---
name: README.md
description: The judged query set, the harness commands, and what each judgement class does and does not establish.
---

# Reference retrieval evaluation

This directory holds the judged query set behind the retrieval-quality findings in `MEASUREMENTS.md`, and `scripts/evaluate_retrieval.py` is the harness that produces them.

The set is small, honest, and reproducible: 32 queries over 19 passages of one real research corpus, judged by inspection of the extracted text, measured through `ResearchService.search`. It is a starting point that can answer "is hybrid better than BM25 here?", not a benchmark suite.

## The judged set

`ai-and-fetishism-queries.json` (schema version 1) contains two parts.

- `targets` — one entry per judged passage: the source path, the locator, a verbatim `snippet`, the `chunk_id_at_measurement`, and a note on what the passage says. The snippet is an identity key, not a quote: it re-resolves the target after re-ingestion, when chunk IDs and chunk boundaries legitimately change. The source path is matched against the generation manifest's documents, and `document_id` is a fallback, so a target also survives a source being renamed.
- `queries` — 32 queries, each judged relevant to exactly one target, in three classes:
  - `quote` (11): a remembered phrasing of the passage, the way a researcher recalls words they read.
  - `paraphrase` (11): the same 11 targets, asked as a question that avoids the author's vocabulary.
  - `entity` (10): a named person, project, place, or concept, asked about specifically.

The `quote` and `paraphrase` classes deliberately share targets, so the two query styles are compared on identical passages. The harness reports the mean content-word overlap between each query and its target passage, so how lexical each class is checkable rather than a claim.

## Protocol and its limits

Judgments are known-item: one designated relevant chunk per query. That measures **findability of a passage**, and it under-credits a mode that returns a different passage making the same point. The harness therefore reports `doc@k` (target document retrieved) next to chunk-level success, so a chunk-level miss inside the right document is not confused with missing the document entirely.

The judgments are single-annotator and were written from the extracted text of the measured generation. There is no inter-annotator agreement, no pooled recall, no usable-precision boundary grade, and no judgment of the passages the systems returned that were not the target. The set holds no query whose correct answer is that the corpus does not answer it, so abstention is not measured by this protocol and `report.no_answer_support` says so rather than reporting a figure. Pooled judgments — every candidate from every mode judged for relevance, which is what proper recall measurement needs — are open work in `TODO.md`.

The judged set can outlive the corpus it was written from. Target `t12` judges `Hall, Race, Articulation and Societies Structured in Dominance.pdf`, which the reviewer excluded on 2026-09-20 as superseded by the Duke reprint in "STUART HALL, SELECTED WRITINGS ON RACE AND DIFFERENCE.pdf"; the generation behind the current numbers does not hold it, so resolution fails unless the run names it with `--skip-targets t12`. A skip is a reviewer decision recorded on the command line and counted in the report (`evaluated_query_count`) rather than a blanket tolerance: every other target still has to resolve to exactly one chunk, every one of them is attempted before anything is reported, and a skipped target neither fails nor excuses a target that was not skipped. Re-pointing `t12` at the retained reprint or retiring it is a judged-set decision rather than a harness one.

Two further limits apply. Relevance gates can legitimately return fewer than `top_k`, so a miss can mean "rejected by a gate" rather than "ranked low". The harness records two different counts for that reason, and they are not interchangeable: `withheld_candidates` counts passages withheld from the *answer*, after ranking, and `dense_gate` counts dense candidates the pre-fusion cosine gate rejected or the relative margin rescued. A search rejecting thirteen candidates per query reports zero withheld, so a run carrying only the first total says nothing about whether the gate is doing work. The search payload carries the gate's own denominator, `dense_gate.eligible_total`, beside the admitted and rejected counts, and the harness checks that those three account for it. A payload carries a gate block for every method, so a block is not evidence that the gate ran: `dense_gate_ran` and `dense_gate_counts_complete` say whether the method ranked the dense half and whether every count was present, and a BM25 row reports `null` rather than a measured zero.

The numbers also describe the generation the report names, so re-run the harness when the corpus, the extraction policy, or a retrieval default changes.

## What a result list contained, beside how it ranked

A known-item query has one relevant passage, so the ranking metrics cannot see a
change that concerns the rest of the list. Two copies of the same evidence rank
first whichever of them is judged relevant, and a different passage carrying the
same evidence displacing the designated chunk reads as a miss. The report
therefore records what each result list contained, and where the judged passage
was in the pipeline, and changes no metric version 1 published.

Per query, `runs[]` carries:

- `distinct_normalized_texts`, `exact_duplicate_slots`, and `exact_duplicate_groups` — how many different passages the list held, by equality of normalized text as an ordered string. Normalization collapses runs of whitespace to one space and trims the ends, and changes nothing else: letter case, word order, a repeated word, a sign, a decimal separator, an operator, a closing mark, and a script with no spaces in it are all differences. `5!` is not `5`, and `V` is not `v`, because a factorial and a variable's case are part of the claim and collapsing either would report a duplicate the corpus does not hold. Two passages differing only in wrapping are one passage.
- `lexical_containment_slots`, `lexical_containment_groups`, and `lexical_containment_pairs` — slots whose ordered word shingles are at least `report.lexical_containment.threshold` present in another returned passage's. It is a lexical overlap, not evidence that two passages make the same point, and it refuses any pair whose numbers, operators, or negations differ. Only the first passage of an exact group is compared, so a reprint is counted once as a reprint and never again as a repetition of itself. Because the exact measure is strict about case and punctuation, this is where a reprint that punctuates or capitalizes differently is found, and both measures read low rather than high.
- `same_source_pairs` — slots sharing a source file with another slot. Distinct evidence often comes from one file, so this is a source-spread measure rather than a duplication one.
- `duplicate_measure_coverage` — `slots_total`, `slots_with_text`, `slots_missing_text`, `missing_chunk_ids`, and `complete`. A returned passage the generation does not hold is missing coverage: the four duplication measures are then `null` and the console prints a warning, because two passages the harness cannot read are not two copies of one passage.
- `dense_gate_ran`, `dense_gate_counts_complete`, `dense_eligible_total`, `dense_admitted_above_floor`, `dense_admitted_below_floor`, `dense_rejected_below_floor`, `dense_quality_excluded`, and `dense_conserved` — the pre-fusion cosine gate's own counts, read from the search payload rather than inferred from the fused list. `null` for a mode that never opens the gate.
- `target_rejected_by` and `target_listed_as_withheld` — which gate named the judged passage, and whether it was withheld from the answer for corrupt text. The examples the engine reports are bounded, so an empty `target_rejected_by` is not proof the target passed every gate.
- `collapsed_count`, `collapsed_by_same_words`, `collapsed_by_same_meaning`, and `collapsed_pairs` — what the repetition collapse removed after reranking, with the pair naming each discarded passage's source and the passage it repeated.
- `candidate_depth`, `candidate_count`, `rerank_requested`, `reranked`, `rerank_fallback`, `rerank_window`, and `relevance_limited` — the branch the search actually took.
- `target_id`, `family_id`, and `target_stage_presence` — the judged passage's target, the question family it belongs to, and whether it was present at each pipeline stage: `dense_before_filters`, `dense_eligible`, `dense_admitted`, `bm25_after_gates`, `fused_pre_rerank`, `reranked`, `post_collapse`, and `final`. A stage the method never opened, and a stage whose identifier list the engine cut before the target's place in it, read as `null`: a target absent from a truncated list was not necessarily absent from the stage.
- `evaluation_trace` — the pipeline's own account, kept whole: every stage's identifier list with its count and truncation flag, the cosine gate's counters, the reranked window and the identifiers it scored, and what the collapse discarded with each discard's source and representative. Identifiers and counts only, no passage text, every list bounded by the engine at 256 with its truncation flag beside it, so a finished report can be audited or intersected without re-running the search. `null` where the engine emitted no trace.
- `reranked_then_collapsed_count`, `reranked_then_collapsed_is_lower_bound`, `reranked_then_collapsed_by_reason`, `collapse_discarded_count`, and `scored_candidate_count` — the reranked window is a budget, and the collapse runs afterwards over both the window and the tail, so these count the candidates the cross-encoder actually scored that were then collapsed away. That is what a wider window would have spent a score on. The discarded side is complete; where the engine's budget cut the scored list the count is a lower bound and the flag says so. The reasons are the engine's own `collapsed_by` labels, reported rather than renamed, and no claim of duplication is made here.

A target ranked fifth and a target the cosine gate dropped are the same number in
every quality column and are not the same event, which is what the stage record
is for. A run also asks the service for the pipeline's own `evaluation_trace`,
which carries those stage identifiers and no passage text and changes no ranking
decision.

Per mode, `summary[mode].overall` adds the means of those per-query measures,
`p50_seconds`, `p95_seconds`, and `max_seconds`, and `summary[mode]` adds
`repeated_slot_rate_all_queries` and `repeated_slot_rate_cross_target_family`.

The repetition rates are measured across the run because no per-query metric can
see a generic leader: a passage that answers every question looks like a
reasonable list in front of each query that returned it. Both rates count each
distinct query once, so one list returning a passage twice is not repetition
across queries. The all-queries rate counts every passage more than one query
returned. The cross-target-family rate counts only passages that answered queries
from more than one question family, so a quote query and its own paraphrase
sharing a target does not register as a generic leader.

A family is the group the judged set's declarations form, not one declaration
chosen over another. A target declares a `family_id` of its own and its queries
may declare others; a target is grouped with every family either declares, and a
target declaring nothing is its own family. A family label and a target id are
separate namespaces, so a label that happens to read like a target id joins
nothing. A run naming no family leaves that rate `null` rather than treating the
query as unrelated to everything.

Latency percentiles are nearest-rank, so every figure is a query the harness
observed. `p95_seconds` is a percentile and not the slowest query: over thirty-two
queries it is the thirty-first, and `max_seconds` is the slowest.

The report carries `schema_version: 3`, `metric_definitions` for every name it
reports, `lexical_containment` naming the threshold, the shingle size, the
denominator, and the distinctions a pair must keep, `superseded_metrics` naming
each report version 2 key and why it is not comparable, `harness` carrying this
script's digest and the checkout's revision, and
`settings.retrieval_policy_fingerprint` beside the resolved retrieval settings. A
measure that was not taken is `null` and prints as a blank column, because a zero
for an absent measure reads as an absence of duplication rather than as an
absence of measurement.

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

Defaults: modes `bm25,dense,hybrid,hybrid+rerank` at `top_k=10`, plus a deep pass for `hybrid` at `top_k=50`, all 32 queries. `--deep-top-k 0` skips the deep pass. `--reranker-model NAME` may be repeated, and the reranked row is then measured once per model over the same queries in one run — which is how `MEASUREMENTS.md` compares the shipped default against the alternative. Without the option the row measures the model the app is configured with (`--reranker-model` or `RESEARCH_RAG_RERANKER_MODEL`, default `Xenova/ms-marco-MiniLM-L-6-v2`). `--offline` works when the runtime and both model caches are already present. Every mode passes `rerank` explicitly, so these numbers do not depend on the app's default; reranking is the default, so the `hybrid+rerank` row is what an ordinary search returns.

Target resolution attempts every target the run did not skip and records every
failure before reporting any: one run names each target that did not resolve
uniquely with its own cause, so a reviewer sees the whole list of decisions a
stale judged set needs instead of one per attempt. The causes are kept distinct
because the remedies are: a source the generation does not hold, two documents
answering one path, a document the generation holds no chunks for, a snippet the
extraction no longer produces, and a snippet two passages both contain. Nothing is
searched when any target failed, because a run that searched the subset that
resolved would report a protocol over fewer judgments than the set declares. A
target named by `--skip-targets` is never attempted, and a skip does not excuse a
failure in a target that was not skipped.

The harness never writes inside the project. It calls `status` and `search` in process through the same `ResearchService` the three surfaces share, and reads the generation's canonical `chunks.jsonl` and `manifest.json` **read-only** to resolve judged targets; that read is measurement-only and not part of the retrieval path. Searches pass `include_staleness=false`, so no result in the report depends on a freshness verdict.

## Output

The console prints two aligned tables (primary depth and deep pass) with, per mode and per class: `succ@1`, `succ@3`, `succ@k`, `MRR`, `nDCG@k`, `doc@k`, mean query-to-target overlap, mean returned passages, and `srcs` — the mean number of distinct sources those passages come from. `srcs` is what a source-diversity reordering is expected to move, so it is reported beside the quality columns rather than instead of them.

Per mode, the table then carries `texts`, `dup`, `lex`, `1src`, `repXF`, `repAll`, `p50s`, `p95s`, and `rej`: the mean distinct normalized texts, mean exact duplicate slots, mean lexical containment slots, mean same-source pairs, the share of slots held by a passage answering more than one family, the same share across all queries, the median and 95th-percentile query in seconds, and the mean dense candidates the cosine gate rejected. A column is left blank where the run carries no such measure. The per-class rows stay on the version 1 columns, because a class of two or three queries cannot support a redundancy mean worth reading.

A reranked row whose cross-encoder did not run is an unranked candidate order, and scoring one under a reranked row's name would be a wrong row. The run fails and names the query unless `--allow-degraded-rerank` is passed, which records every affected run under `report.degraded` and prints them as blanks rather than averaging them in. A hybrid row the payload shows without a fusion block fails the same way.

A full JSON report is written beside the judged set as `ai-and-fetishism-queries-report.json` with every per-query run, the resolved targets, the retrieval configuration recorded in the generation, the timing, and the run's own settings — the modes measured, the reranker models, `evaluated_query_count`, `skipped_targets`, `selection_policy`, which names the source-diversity penalty the run used because the generation's recorded policy cannot carry a value applied after ranking, and `retrieval_policy_fingerprint` with the resolved retrieval values the run cannot be reproduced without. Reports are generated artifacts, ignored by git; regenerate one instead of editing it.

## Extending the set

Add targets by copying an exact snippet out of the corpus text and confirming that `--validate-only` resolves it to exactly one chunk. Ambiguity is refused rather than guessed, so a snippet that spans a chunk overlap or appears twice in a document must be extended or replaced before it can be judged. Adding queries only requires a `query_id`, a `class`, a `target_id`, and the query text.

Authorization over the corpus is unchanged by this directory: the original PDF and EPUB files remain the quote authority, and the snippets here are short identity keys used to resolve a judged passage.
