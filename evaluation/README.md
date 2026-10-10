# Reference retrieval evaluation

- The judged query set is 32 queries over 19 passages of one research corpus. Judgments come from inspecting the extracted text. Measurement runs through `ResearchService.search`.
- It answers "is hybrid better than BM25 here?". It is not a benchmark suite.
- The numbers describe the generation the report names. Re-run the harness when the corpus, extraction policy, or a retrieval default changes.

## The judged set

`ai-and-fetishism-queries.json` (schema version 1) has two parts.

- `targets`: one entry per judged passage.
  - Each holds the source path, the locator, a verbatim `snippet`, the `chunk_id_at_measurement`, and a note on what the passage says.
  - The snippet is an identity key, not a quote. It re-resolves the target after re-ingestion, when chunk IDs and boundaries change.
  - The source path matches the generation manifest's documents. `document_id` is a fallback, so a target survives a source rename.
- `queries`: 32 queries, each judged relevant to exactly one target.
  - `quote` (11): a remembered phrasing of the passage.
  - `paraphrase` (11): the same 11 targets, asked without the author's vocabulary.
  - `entity` (10): a named person, project, place, or concept.
- `quote` and `paraphrase` share targets, so both styles are compared on identical passages. The harness reports the mean content-word overlap between each query and its target.

## Protocol

- Judgments are known-item: one designated chunk per query. That measures findability of a passage. `MEASUREMENTS.md` lists what this cannot establish.
- The harness reports `doc@k` (target document retrieved) beside chunk-level success, so a miss inside the right document is not confused with missing the document.
- The set holds no query whose correct answer is "the corpus does not answer this". Abstention is unmeasured, and `report.no_answer_support` says so instead of reporting a figure.
- Pooled judgments, which proper recall needs, are open in `TODO.md`.
- Target `t12` judges `Hall, Race, Articulation and Societies Structured in Dominance.pdf`.
  - The reviewer excluded it on 2026-09-20 as superseded by the Duke reprint in "STUART HALL, SELECTED WRITINGS ON RACE AND DIFFERENCE.pdf".
  - The generation behind the current numbers does not hold it, so resolution fails unless the run passes `--skip-targets t12`.
  - A skip is a reviewer decision recorded on the command line and counted in `evaluated_query_count`. Every other target must still resolve to exactly one chunk, and a skip never excuses a target that was not skipped.
  - Re-pointing `t12` at the retained reprint, or retiring it, is a judged-set decision.
- Relevance gates can return fewer than `top_k`, so a miss can mean "rejected by a gate" and not "ranked low". Two counts record this, and they are not interchangeable:
  - `withheld_candidates` counts passages withheld from the answer after ranking.
  - `dense_gate` counts dense candidates the pre-fusion cosine gate rejected or the relative margin rescued.
  - A search rejecting thirteen candidates per query can report zero withheld.
  - The payload's `dense_gate.eligible_total` is the gate's denominator, and the harness checks that the admitted and rejected counts account for it.
  - Every method's payload carries a gate block, so a block does not show the gate ran. `dense_gate_ran` says whether the method ranked the dense half. `dense_gate_counts_complete` says whether every count was present. A BM25 row reports `null`, not a measured zero.

## What the report records beside rank

- Known-item ranking cannot see a change to the rest of the list. Two copies of one passage rank first whichever is judged relevant, and a different passage carrying the same evidence reads as a miss.
- The report records what each result list contained and where the judged passage was in the pipeline. Metric version 1 is unchanged.
- A measure that was not taken is `null` and prints blank. A zero would read as an absence of duplication, not an absence of measurement.
- Per query, `runs[]` carries:
  - **Exact duplicates:** `distinct_normalized_texts`, `exact_duplicate_slots`, and `exact_duplicate_groups`.
    - Normalization collapses whitespace runs to one space and trims the ends. Case, word order, repeated words, signs, decimal separators, operators, closing marks, and scripts without spaces are differences. `5!` is not `5`, and `V` is not `v`.
    - Two passages differing only in wrapping are one passage.
  - **Lexical containment:** `lexical_containment_slots`, `lexical_containment_groups`, and `lexical_containment_pairs`.
    - A slot counts when its ordered word shingles are at least `report.lexical_containment.threshold` present in another returned passage.
    - It is lexical overlap, not proof of the same point. It refuses any pair whose numbers, operators, or negations differ.
    - Only the first passage of an exact group is compared, so a reprint counts once.
    - This is where a reprint that punctuates or capitalizes differently is found.
    - Both duplicate measures read low, not high.
  - **Source spread:** `same_source_pairs` counts slots sharing a source file. It measures spread, not duplication.
  - **Coverage:** `duplicate_measure_coverage` holds `slots_total`, `slots_with_text`, `slots_missing_text`, `missing_chunk_ids`, and `complete`. A returned passage the generation does not hold makes the four duplication measures `null` and prints a warning.
  - **Dense gate:** `dense_gate_ran`, `dense_gate_counts_complete`, `dense_eligible_total`, `dense_admitted_above_floor`, `dense_admitted_below_floor`, `dense_rejected_below_floor`, `dense_quality_excluded`, and `dense_conserved`. They come from the payload, not the fused list. `null` for a mode that never opens the gate.
  - **Target rejection:** `target_rejected_by` and `target_listed_as_withheld`. The engine's examples are bounded, so an empty `target_rejected_by` does not prove the target passed every gate.
  - **Collapse:** `collapsed_count`, `collapsed_by_same_words`, `collapsed_by_same_meaning`, and `collapsed_pairs`, which names each discarded passage's source and the passage it repeated.
  - **Branch taken:** `candidate_depth`, `candidate_count`, `rerank_requested`, `reranked`, `rerank_fallback`, `rerank_window`, and `relevance_limited`.
  - **Target stage:** `target_id`, `family_id`, and `target_stage_presence`.
    - Stages: `dense_before_filters`, `dense_eligible`, `dense_admitted`, `bm25_after_gates`, `fused_pre_rerank`, `reranked`, `post_collapse`, and `final`.
    - A stage the method never opened reads `null`. So does a stage whose identifier list the engine cut before the target's place in it, because absence from a truncated list is not absence from the stage.
    - A target ranked fifth and a target the cosine gate dropped share a rank column and are not the same event. The stage record separates them.
  - **Trace:** `evaluation_trace` is the pipeline's own account, kept whole. It holds every stage's identifier list with its count and truncation flag, the cosine gate's counters, the reranked window and scored identifiers, and what the collapse discarded with each source and representative. It holds identifiers and counts only, no passage text. The engine bounds every list at 256 with its truncation flag. It is `null` where the engine emitted no trace.
  - **Scored then collapsed:** `reranked_then_collapsed_count`, `reranked_then_collapsed_is_lower_bound`, `reranked_then_collapsed_by_reason`, `collapse_discarded_count`, and `scored_candidate_count`.
    - These count candidates the cross-encoder scored and the collapse then removed, which is what a wider window would have spent a score on.
    - The discarded side is complete. Where the engine's budget cut the scored list, the count is a lower bound and the flag says so.
    - Reasons are the engine's own `collapsed_by` labels. No duplication claim is made.
- Per mode, `summary[mode].overall` adds the means of those measures, `p50_seconds`, `p95_seconds`, and `max_seconds`. `summary[mode]` adds `repeated_slot_rate_all_queries` and `repeated_slot_rate_cross_target_family`.
  - No per-query metric can see a generic leader, because a passage that answers every question looks reasonable in front of each query. Both rates are measured across the run and count each distinct query once.
  - The all-queries rate counts every passage more than one query returned.
  - The cross-family rate counts only passages that answered queries from more than one question family. A quote query and its paraphrase sharing a target does not count.
  - A family is the group the set's declarations form. A target joins every family it or its queries declare, and a target declaring none is its own family. Family labels and target IDs are separate namespaces. A run naming no family leaves the rate `null`.
- Latency percentiles are nearest-rank, so every figure is a query the harness observed. Over 32 queries `p95_seconds` is the thirty-first, and `max_seconds` is the slowest.
- The report carries `schema_version: 3` and:
  - `metric_definitions` for every reported name.
  - `lexical_containment`, naming the threshold, shingle size, denominator, and the distinctions a pair must keep.
  - `superseded_metrics`, naming each version 2 key and why it is not comparable.
  - `harness`, with the script's digest and the checkout's revision.
  - `settings.retrieval_policy_fingerprint` beside the resolved retrieval settings.

## Running it

```bash
uv run python scripts/evaluate_retrieval.py --project /path/to/project
uv run python scripts/evaluate_retrieval.py --project /path/to/project --validate-only
uv run python scripts/evaluate_retrieval.py --project /path/to/project --limit 5 --modes bm25,hybrid
uv run python scripts/evaluate_retrieval.py --project /path/to/project --deep-top-k 0 \
    --modes hybrid,hybrid+rerank \
    --reranker-model Xenova/ms-marco-MiniLM-L-6-v2 \
    --reranker-model jinaai/jina-reranker-v1-turbo-en
```

- Defaults: modes `bm25,dense,hybrid,hybrid+rerank` at `top_k=10`, a deep pass for `hybrid` at `top_k=50`, and all 32 queries.
- `--deep-top-k 0` skips the deep pass.
- `--reranker-model NAME` can repeat. The reranked row is then measured once per model over the same queries.
  - Without it, the row uses the configured model: `--reranker-model` or `RESEARCH_RAG_RERANKER_MODEL`, default `Xenova/ms-marco-MiniLM-L-6-v2`.
- `--offline` works when the pinned model caches and GPT-2 tokenizer cache are present. The harness uses the direct backend and needs no legacy gateway runtime.
- Every mode passes `rerank` explicitly, so numbers do not depend on the app's default. The `hybrid+rerank` row is what an ordinary search returns.
- Target resolution attempts every target not skipped and records every failure before reporting any. Each failure names its own cause, so a reviewer sees every decision a stale set needs. The causes are:
  - A source the generation does not hold.
  - Two documents answering one path.
  - A document with no chunks in the generation.
  - A snippet the extraction no longer produces.
  - A snippet two passages both contain.
- Nothing is searched when any target failed, because a run over the resolved subset would report a protocol over fewer judgments than the set declares.
- The harness never writes inside the project.
  - It calls `status` and `search` in process through the shared `ResearchService`.
  - It reads the generation's `chunks.jsonl` and `manifest.json` read-only to resolve targets. That read is measurement-only and not part of the retrieval path.
  - Searches pass `include_staleness=false`, so no result depends on a freshness verdict.

## Output

- The console prints two aligned tables, primary depth and deep pass. Per mode and class they show: `succ@1`, `succ@3`, `succ@k`, `MRR`, `nDCG@k`, `doc@k`, mean query-to-target overlap, mean returned passages, and `srcs`.
  - `srcs` is the mean number of distinct sources, which a source-diversity reordering is expected to move.
- Per mode the table adds `texts`, `dup`, `lex`, `1src`, `repXF`, `repAll`, `p50s`, `p95s`, and `rej`.
  - `texts`: mean distinct normalized texts.
  - `dup`: mean exact duplicate slots.
  - `lex`: mean lexical containment slots.
  - `1src`: mean same-source pairs.
  - `repXF`: share of slots held by a passage answering more than one family.
  - `repAll`: the same share across all queries.
  - `p50s`, `p95s`: median and 95th-percentile query time in seconds.
  - `rej`: mean dense candidates the cosine gate rejected.
  - A column is blank where the run has no such measure. Per-class rows keep the version 1 columns, because two or three queries cannot support a redundancy mean.
- A reranked row whose cross-encoder did not run is an unranked order, and scoring it under a reranked name would be a wrong row.
  - The run fails and names the query unless `--allow-degraded-rerank` is passed. That flag records every affected run under `report.degraded` and prints them blank instead of averaging them in.
  - A hybrid row whose payload has no fusion block fails the same way.
- A full JSON report is written beside the judged set as `ai-and-fetishism-queries-report.json`. It holds:
  - Every per-query run, the resolved targets, the retrieval configuration recorded in the generation, and the timing.
  - The run's settings: the modes, reranker models, `evaluated_query_count`, `skipped_targets`, and `selection_policy`.
  - `selection_policy` names the source-diversity penalty used, because the generation's recorded policy cannot carry a value applied after ranking.
  - `retrieval_policy_fingerprint`, with the resolved retrieval values needed to reproduce the run.
- Reports are generated artifacts ignored by git. Regenerate one instead of editing it.

## Extending the set

- Add a target by copying an exact snippet from the corpus text and confirming that `--validate-only` resolves it to exactly one chunk.
  - Ambiguity is refused. A snippet that spans a chunk overlap or appears twice in a document must be extended or replaced.
- Add a query with a `query_id`, a `class`, a `target_id`, and the query text.
- The original PDF and EPUB files remain the quote authority. The snippets are short identity keys.
