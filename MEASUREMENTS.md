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

## Current reranker screening (2026-10-10)

- Protocol: `evaluation/README.md`; schema-3 offline reports `/tmp/kilo/reranker-20261010-offline-pass-{1,2,3}.json`, harness revision `fef9fadd0e8e1586c8cb1efa56de88aacc45450d`, Python 3.12.3. The initial `pass-.json` provisioning artifact is excluded. Private reports remain outside Git.
- The reports' live `project` block identifies generation `20261009T201931Z-66ebc96e`, 39,298 chunks and 132 indexed sources, with no generation upgrade required. Their `corpus` block describes the older judged-set reference, not the searched generation. The reports supply no corpus content hash.
- Thirty unique known-item queries cover 18 targets. Explicit `--skip-targets t12` excludes the absent Hall source under the evaluation protocol. Three passes of five conditions produce 450 repeated searches, not independent samples. Reply depth is ten; no deep pass runs; staleness checks are disabled.
- Every condition observes branch depth 40. All 360 requested reranked searches apply reranking without fallback and score exactly 20 candidates. Ordered stage identifiers and returned identifiers match across passes for each query and condition.
- Models use the CPU FastEmbed ONNX cross-encoder. Revision pins are owned by `src/research_rag/retrieval/rerankers.py::RERANKER_MODELS` at the recorded revision: L6 `a09144355adeed5f58c8ed011d209bf8ee5a1fec`, L12 `42a4a787e30451cf9dbd09080c2a5b8dde332c1e`, Jina tiny `aca45de6945b5dc6399abcd2a9c55ded5dc9111f`, Jina turbo `b8c14f4e723d9e0aab4732a7b7b93741eeeb77c2`. L6/L12 denote `Xenova/ms-marco-MiniLM-L-{6,12}-v2`; Jina tiny/turbo denote `jinaai/jina-reranker-v1-{tiny,turbo}-en`. The dense report records `qdrant/bge-small-en-v1.5-onnx-q` revision `aa8f8b060edb00e03bfdd08813a2949946c8ba55`.

Quality is identical in all three passes. Counts retain the denominator of 30; MRR and binary-gain nDCG use the designated chunk only.

| Condition | Success@1 | Success@3 | Success@10 | Document@10 | MRR | nDCG@10 |
|---|---:|---:|---:|---:|---:|---:|
| Hybrid, no reranker | 19/30 | 21/30 | 25/30 | 27/30 | 0.686429 | 0.721267 |
| Hybrid + L6 | 24/30 | 25/30 | 26/30 | 27/30 | 0.821429 | 0.832142 |
| Hybrid + L12 | 24/30 | 26/30 | 26/30 | 27/30 | 0.833333 | 0.842062 |
| Hybrid + Jina tiny | 23/30 | 23/30 | 26/30 | 27/30 | 0.787222 | 0.805791 |
| Hybrid + Jina turbo | 22/30 | 24/30 | 25/30 | 27/30 | 0.769444 | 0.785387 |

Latency entries are seconds, median/p95 per pass, using nearest-rank percentiles over 29 searches after excluding `q01` separately in each condition. The last column gives the excluded first-use `q01` range across passes.

| Condition | Pass 1 | Pass 2 | Pass 3 | First-use range |
|---|---:|---:|---:|---:|
| Hybrid, no reranker | 0.360/0.625 | 0.192/0.498 | 0.212/0.525 | 0.500–0.640 |
| Hybrid + L6 | 1.497/1.852 | 1.410/1.652 | 1.419/1.620 | 1.647–2.059 |
| Hybrid + L12 | 2.644/3.530 | 2.618/2.938 | 2.620/2.905 | 3.014–3.741 |
| Hybrid + Jina tiny | 1.263/1.595 | 1.255/1.466 | 1.278/1.362 | 1.640–2.043 |
| Hybrid + Jina turbo | 1.789/2.304 | 1.750/1.935 | 1.772/1.917 | 2.279–2.851 |

- The discarded warm-up runs BM25 only (9.643–11.546 seconds); it does not warm dense inference or each cross-encoder. First-use entries are measured searches, not isolated model-load times. Fixed condition order and three small repeated passes do not establish a stable population p95 or a performance winner.
- Against L6, L12 regresses `q06` from rank 1 to 2 and improves `q27` from 7 to 1. Jina tiny regresses `q15` from 1 to 5 and `q29` from 1 to 4; it improves `q25` from 2 to 1 and `q27` from 7 to 6. Jina turbo loses `q06` from the top ten and regresses `q16` from 1 to 4; it improves `q27` from 7 to 3.
- All five conditions miss `q14`, `q22`, `q17`, and `q18`. The first two targets are outside the fused candidate pool; the latter two reach that pool but lie outside the 20-candidate scored window. A reranker comparison at this window cannot repair those upstream misses.
- This is screening on one inspected, single-annotator known-item set, not pooled relevance, exhaustive recall, abstention, or held-out confirmation. Correlated target families and repeated searches do not establish model superiority. Defaults, judgments, sources, and generation state remain unchanged.

## Descriptive paraphrase experiment (2026-10-10)

- Private records: `/tmp/kilo/paraphrase-20261010/{report,summary,stage-analysis,fidelity-review,variants}.json`. Query and passage text remain outside Git.
- Report identity: 80 cells, eight conditions and ten shared targets; fixed generation `20261009T201931Z-66ebc96e`, hybrid + `Xenova/ms-marco-MiniLM-L-6-v2` (L6), retrieval-policy fingerprint `b7285824c79d662c9abdba84238077abc1271d11a861c931ee0e590c3d4f380f`. The offline run requests ten results, records no searches, and uses order seed `20261010`. All cells apply reranking without fallback; generation identity stays fixed.
- Exact is a canonical cleaned-text excerpt control, not a verified source quotation. Format uppercases exact; syntax and lexical manipulate excerpts; structural and conceptual are questions. Context additions, question form and answer slots confound these comparisons. They do not estimate a causal effect of paraphrase depth.
- Historical baseline questions remain unchanged despite documented scope, modality, attribution and target-note drift. They are historical comparators, not strictly equivalent controls.
- Anchored appends source keyword anchors to the conceptual question as query text, not metadata filters. These privileged, target-informed additions can disclose answer names or source entities; they do not test ordinary unassisted question augmentation.

Counts and MRR credit only the designated target chunk in the final top ten, with reciprocal rank zero for a miss. Each condition has ten correlated target observations.

| Condition | Target@1 | Target@10 | MRR |
|---|---:|---:|---:|
| Historical baseline | 6/10 | 6/10 | 0.60 |
| Format | 10/10 | 10/10 | 1.00 |
| Syntax | 10/10 | 10/10 | 1.00 |
| Lexical | 8/10 | 9/10 | 0.85 |
| Structural | 9/10 | 10/10 | 0.95 |
| Conceptual | 5/10 | 6/10 | 0.55 |
| Anchored | 7/10 | 9/10 | 0.80 |
| Exact canonical text | 10/10 | 10/10 | 1.00 |

- Paired conceptual-to-anchored target recovery rises from 6/10 to 9/10: `t03`, `t11` and `t15` enter the top ten. MRR rises from 0.55 to 0.80. Designated-target recovery does not establish full relevance, usable evidence, or exhaustive recall. Answer-bearing neighbours can score as target misses.
- A separate model reviewer saved blinded passage assessments before reading condition labels. The rubric distinguishes full support, partial support, topical evidence and irrelevant evidence, preserving the requested relationship and modality. These are model assessments of cleaned text, not human ground truth or verified quotations; `judgments.json` retains passage IDs, ranks, reasons and ambiguity flags.
- Model-assessed full support rises from 6/10 to 8/10 at rank one and from 8/10 to 10/10 somewhere in the returned ten for conceptual versus anchored questions. Best-of-ten means one supporting passage, not synthesis across passages. Anchors repair the recycling and writer-compensation evidence gaps; the financial-elites and model-collapse questions already return fully supporting alternatives without their designated targets. Four of the ten designated-target misses across all conditions have full alternative evidence.
- Observed branch candidate depth is 40 and the scored window is 20; fused pools contain 40–80 candidates. Baseline `t08` and `t11` reach fusion at ranks 63 and 64, and conceptual `t03` at rank 34, outside the scored window. Other misses never reach fusion. These traces describe losses, not a causal depth sweep or proof that widening a window repairs them.
- Lexical `t03` reaches dense eligible rank 3 but fails the dense floor (`dense_below_threshold`), has no admitted lexical target, and never reaches fusion. Anchors recover `t08` to the fused pool at rank 10, but reranking places it at rank 15, outside the final ten. The scored stage excludes the unscored tail; a final omission is not automatically a gate rejection.

## Critical-realism preprocessing experiment (2026-10-10)

- Private records live under `/tmp/kilo/critical-realism-preprocessing-20261010/`: `paired-diagnostics-costs.json`, `paired-anchor-retention.json`, `paired-negative-controls.json`, `paired-source-integrity.json`, `{custom,docling}-setup.json`, and `paired-retrieval-evaluation/{summary,protocol,artifact-integrity}.json`. `paired-analysis-provenance.json` records exact input and analysis-script SHA-256 identities; setup and integrity records retain per-source hashes. Private text and reports remain outside Git.
- Both isolated private arms index the same 17 sources: 13 PDF, three EPUB and one MOBI. The byte-identical Naturalism EPUB duplicate is retained in both arms. Original before/after hashes, snapshot hashes and manifest hashes agree. No original source changes.
- Effective settings differ only in `pdf_backend`; both use recorded defaults with two embedding threads, chunk size/overlap 384/64 and headers off. Dense inference uses `qdrant/bge-small-en-v1.5-onnx-q@aa8f8b060edb00e03bfdd08813a2949946c8ba55`; reranking uses `Xenova/ms-marco-MiniLM-L-6-v2@a09144355adeed5f58c8ed011d209bf8ee5a1fec`. Setup records retain installed packages and Docling runtime pins (`docling==2.135.0`, `rapidocr==3.9.1`, `pymupdf>=1.26,<2`).
- Checkout base is `fef9fadd0e8e1586c8cb1efa56de88aacc45450d`, with uncommitted MOBI, extraction and other changes. Git HEAD alone does not reproduce the executed code; the retained records do not establish a complete immutable dirty-checkout snapshot.

| Arm | Generation | Extraction units | Chunks |
|---|---|---:|---:|
| Custom | `20261010T110139Z-f988ef97` | 9,675 | 12,210 |
| Docling | `20261010T112412Z-67b31671` | 21,554 | 21,976 |

- Both generations are ready and non-partial. Unit, chunk and loss counts describe pipeline output, not source fidelity.
- The model-prepared, original-verified sample contains 17 anchors. Each arm retains all 17 anchors' normalized tokens in order; 16 occur contiguously. Protected negations survive. These lexical diagnostics establish neither full-page retention nor visual reading order. No anchor contains numeric tokens, so number retention is untested.
- All 7,029 EPUB/MOBI negative-control units are exactly equal in ordered extraction-bearing payloads, without text normalization and excluding only `id`, `document_id` and `source_id`.
- Custom recorded ingestion time is 1,352.6 seconds. Docling's start-to-continuation-end wall span is 7,967.5 seconds, including an unquantified interruption gap and a 2,798.3-second continuation; it is not uninterrupted processing time. Provisioning takes 87.2 seconds separately. The handoff span already includes continuation and must not be added to it. Recorded cumulative RSS maxima do not support a valid per-arm memory comparison.
- Retrieval runs offline with ten results and no search-history writes: 40 conditions per arm, ten shared needs and four variants (baseline, lexical, structural, conceptual). All 80 searches pass strict pipeline validation; arm artifacts and original sources remain unchanged. All conditions, including misses, remain in the denominator. Anchor matching uses the protocol's normalization without paraphrase matching; these are known-item diagnostics, not pooled relevance or exhaustive recall.

| Arm | Anchor@1 | Anchor@10 | Anchor MRR@10 | Source@10 | Physical page@10 |
|---|---:|---:|---:|---:|---:|
| Custom | 19/40 | 28/40 | 0.5306 | 37/40 | 28/40 |
| Docling | 25/40 | 29/40 | 0.6558 | 36/40 | 29/40 |

- Independent model evidence judgments were saved before joining the arm mapping: `paired-retrieval-evaluation/{evidence-judgments,evidence-summary-private}.json`. The rubric scores full support 3, partial support 2, topical evidence 1 and irrelevant evidence 0, including required qualifications. These are model assessments, not human annotations. Best-of-ten scores one passage, never combined passages.

| Arm | Rank-one full | Rank-one partial-only | Best single top-ten full | Best single top-ten partial-only |
|---|---:|---:|---:|---:|
| Custom | 20/40 | 7/40 | 28/40 | 6/40 |
| Docling | 22/40 | 9/40 | 26/40 | 10/40 |

- The review identifies mixed tradeoffs: custom retains the complete Donati claim in one passage where Docling splits it; Docling returns the required Porpora qualifier at rank one. Supporting and ambiguous candidates received full reads; other candidates were screened through beginnings and need-bearing clauses. The reviewer used supplied model-verified original-source context as the reference and opened no additional source images. This is not a new original-source verification.
- Forty variants are correlated within ten information-need families, not independent trials or evidence of statistical significance. These retrieval assessments establish neither overall source quality nor global superiority.
- The separate blinded page-fidelity model review inspected all 17 supplied full-page renders and saved `preprocessing-fidelity-judgments.json` before reading the per-page arm mapping. Preferences are custom four, Docling two, mixed eight and tie three; these are ordinal model judgments, not human annotations or retrieval scores.
- Custom shows advantages in visible opening coverage, title order and a causal contrast. Docling shows advantages in footnote qualifications, author attribution and column order. Both have diagram-structure defects, a footnote-interrupted sentence and a chemical-formula letter-to-digit substitution. Central negations survive; no differential negation reversal was observed.
- Findings concern serialized page packets, not proven exclusion from retrievable evidence. Off-page unit continuations remain unverified except across the supplied consecutive Gorski pages; no exhaustive character transcription was performed. These qualitative mechanism tradeoffs establish no overall source-quality winner. Defaults remain unchanged.
