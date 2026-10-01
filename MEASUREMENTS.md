# Measurements

Every number here is a current, reproducible property of this app and the workload it targets. No past state of the code: history lives in git. Two harnesses produce all of it, each driving the real gateway, the real models, and the real service:

```bash
uv run python scripts/benchmark_write_pattern.py --root /path/on/target/disk
uv run python scripts/evaluate_retrieval.py --project /path/to/project --offline
```

The retrieval numbers were measured against the pinned engine this app runs, and the service layer is that project's unchanged code. The engine and its payloads are the previous ones: the same seven operations behind the same lean/full projection, so every payload figure below is that project's.

## 1. Workload and design envelope

Built and measured for 25–250 English-primary born-digital PDF/EPUB sources, 5,000–50,000 chunks, on CPU only, with humanities and social-science prose that carries footnotes, reference lists, tables, and short quotations in other scripts. Out of scope, and not claimed: non-English-primary corpora, multilingual retrieval, OCR or scanned material, handwriting, and formula-heavy corpora.

### Reference corpus

| Property | Value |
|---|---|
| Sources | 57 discovered, 55 indexed, 2 reviewed exclusions |
| Source material | 135 MB of PDF and EPUB originals |
| Documents | 55 |
| Extraction units | 9,546 |
| Chunks | 13,158 (about 239 per source) |
| Vectors | 13,158 × 384 float32 = 20.2 MB |
| Generation on disk | 68 MB in 14 files |
| Generation schema | 5 |
| Dense index | portable float32 vectors, exact cosine scan |
| Character | English humanities and social-science prose with footnotes, reference lists, tables, and short quotations in other scripts |

## 2. Ingestion cost

### Full rebuild of the reference corpus on NVMe storage, CPU only

| Phase | Time |
|---|---|
| source hashing | 1.7 s |
| extraction | 281.9 s |
| chunking | 51.3 s |
| embedding (13,158 chunks) | 668.4 s |
| BM25 index | 2.2 s |
| dense index (exact scan) | 0.07 s |
| assembly | 1.3 s |
| **total** | **about 1,007 s (17 minutes)** |

Embedding dominates, and it is CPU-bound: phase timings are only weakly device-dependent. The same build reports 8 of 13,158 chunks above the embedding model's 512-token input limit, the largest at 3,417 of that tokenizer's tokens because the chunker counts GPT-2 tokens and the embedding model counts its own. They are counted and flagged per chunk rather than split, because splitting would change chunk identities and reuse without a long-input strategy. The build discarded 1 corrupt chunk and excluded 16 corrupt extraction units, each disclosed with its locator and reason.

### What durability costs, and why writes are grouped

One atomic write, 17–35 KB payload, measured on both devices:

| Operation | HDD | NVMe |
|---|---|---|
| write + `os.replace`, no fsync | 0.40 ms | 0.36 ms |
| + file fsync | 33.95 ms | 1.62 ms |
| + directory fsync | 91.25 ms | 2.67 ms |
| 8 artifact files sharing one directory fsync | 803 ms/batch | 14.5 ms/batch |

Payload size is irrelevant on the HDD: a 17 KB checkpoint costs the same as a 35 KB one. Cost follows the number of durability operations, so a unit's artifacts are written together and committed with one directory fsync before the checkpoint that claims the unit complete. Grouping cut chunking by 17–37 s on a 64-unit HDD corpus. `assembly` stayed under 1.7 s in every run, inside the machine's noise, so it is not claimed as an improvement.

### Chunking in batches of 16 extraction units

| Phase | One unit per call | 16 units per call |
|---|---|---|
| chunking | 25.56 s | 15.88 s |
| wall clock | 117.66 s | 86.82 s |

Chunk identities are identical either way. A crash redoes at most one batch of 16 units.

### Why the dense index is an exact scan by default

| Backend | Full rebuild of the reference corpus |
|---|---|
| embedded ANN index (Qdrant local mode) | 3,040.73 s and a 105 MB generation, measured on the project's HDD |
| exact scan of the generation's portable vectors | 0.07 s of index work and a 68 MB generation, on NVMe |

Three things decide this. The embedded build is dominated by per-point device cost: 75.5× slower on that HDD than the same build on NVMe. It returns neither chunk identity nor scores through the upstream API, which the research contract needs. The exact scan reads the portable float32 vectors the generation already stores, so it needs no separate index and returns the same top 20 as a brute-force ranking. Above 200,000 chunks the exact scan stops being the right default and the embedded backend is chosen instead; that threshold is arithmetic from measured exact-scan cost, not a measurement at that size.

### Embedding: the inference batch is a padding decision

FastEmbed pads every sequence to the longest member of its batch, and ONNX Runtime still computes the padded positions, so cost is about `0.27 ms × padded tokens`. Measured on 128 real chunk texts (mean 152 tokens, median 112, p90 330, max 846), 64 texts per configuration:

| Inference batch | 64 texts | chunks/s |
|---|---|---|
| 1 | 2.70 s | **23.68** |
| 4 | 8.13 s | 7.87 |
| 8 | 12.86 s | 4.98 |
| 16 | 12.08 s | 5.30 |
| 32 | 11.07 s | 5.78 |
| 64 | 14.27 s | 4.49 |
| 16, input sorted by length | 6.95 s | 9.21 |
| 64, input sorted by length | 13.31 s | 4.81 |

The token model predicts every row: a batch of one costs `sum(lengths) ≈ 9,728` tokens (2.6 s predicted, 2.70 s measured) while a batch of 64 costs `64 × 846 = 54,144` (14.6 s predicted, 14.27 s measured). Sorting narrows each group and still cannot beat a batch of one, which pads nothing. `EMBEDDING_INFERENCE_BATCH_SIZE` is therefore 1, and the vectors are identical either way: batch 1 and batch 64 returned bit-identical floats (`max |delta| = 0.0`) on two independent 64-text slices.

Threads (ONNX intra-op and inter-op) at batch 1 on the same texts: runtime default 23.65 chunks/s, 1 → 9.31, 4 → 23.97, 8 → **31.66**, 16 → 20.03. The optimum was the physical core count, which is machine-specific, so `--embedding-threads` is unset by default: setting it to 16 measured *worse* than leaving it alone, and nothing is auto-detected.

## 3. Candidate gating and per-query overhead

### The candidate gate

A chunk's usability verdict is a property of the chunk, not of the query, so it is computed once when a generation's artifact lookup is built and stored as a small bitmask per chunk. Measured on 200 real chunk texts, per query:

| Stage | Stored verdict | Recomputed per query |
|---|---|---|
| artifact-lookup fetch including offset parse | 6.78 ms | 6.90 ms |
| corrupt-text scan | at build time | 58.32 ms |
| searchable-content scan | at build time | 13.65 ms |
| extraction-artifact check | at build time | 14.12 ms |
| content-token extraction (query-dependent) | 2.18 ms | 8.37 ms |
| **gate total** | **8.96 ms** | **92.06 ms** |

That is 10.3× cheaper at the 200 candidates a reference-view query can reach, and about 1.4 ms instead of 15 ms at the default depth of 32. One shared function computes the bitmask for both the build and the query-time fallback, so a rejection decision and the counter that reports it cannot drift. Reason codes are recomputed only for a chunk the verdict flags as corrupt, because the response discloses them, and a generation whose lookup predates the column recomputes the verdict from text.

### What a reader waits for

One app process serves the workspace, the agent, and the command line, and it resolves settings and starts answering before the retrieval stack is imported and the gateway is opened, so a project reads fine on a machine where the UltraRAG runtime is not installed yet. Measured on the reference project, from spawning `research-rag` to the first answer:

| What | Cost |
|---|---|
| process start to a `status` answer, no gateway yet | **0.27 s** |
| process start to a workspace, agent surface, and control API, all on one port | **0.41 s** |
| first answer that needs the gateway (`ingest`, or `search` on a built generation) | 2.16 s, once per process |
| first `search`, which loads the embedding and reranking models | **11.37 s**, once per process |
| every later `search` | 1.06 s |
| `sources` | 0.08 s |

A gateway that cannot start is reported by the operation that needed it, never by a surface that never appeared, and the answer carries the tail of the logs the transport wrote plus the path of each one. The first search pays the cold cost of the engines this app runs: a query embedding model and a cross-encoder load from disk on first use, and nothing downloads because the models are cached.

### Fixed overhead per query

Measured on the reference project, medians:

| Operation | Cost |
|---|---|
| source-tree walk (`scan_sources`, the staleness check) | **9.69 ms** for 55 sources |
| rebuilding the effective-document map | 0.615 ms |
| fingerprinting the metadata that would key a cached map | 0.013 ms |
| opening one artifact-lookup SQLite connection | 0.094 ms |
| artifact lookup built from scratch plus one count query | 1.32 ms |
| the same count query on a warm instance | 0.319 ms |
| dependency report, first call in a process (one read of the 11 MB runtime tree) | **46.6 ms** |
| dependency report, every later call (one marker stat) | 2.41 ms |

The dependency report is the one term that reads outside this project: it validates the managed UltraRAG runtime, which hashes 11 MB, so it costs 46.6 ms the first time a process asks and 2.41 ms after that, against a `status` that measures 127 ms on its first call and a 76 ms median after it. It is cached per process and re-read only when the marker identifying the installed runtime changes, which is what a reinstall or a repair rewrites, so the cost is paid once per process rather than once per call. `search` never pays it.

The staleness walk is an order of magnitude larger than everything else and the only term that grows with the number of source files: about 0.18 ms per source, so roughly 176 ms at 1,000 sources. That is why the engine takes a staleness switch at all, and only the measurement harness sets it: no reader-facing operation can skip it, because a search that cannot say whether its corpus moved is answering about something unknown. Nothing else here is cached, deliberately: caching the document map saves about 0.6 ms per query while adding cross-request state that must be invalidated correctly, and reusing SQLite connections saves about 0.09 ms while requiring one connection to be reachable from whichever thread serves the next call. At the scale where either would matter, parsing the manifest dominates both, so the right answer there is a persistent document-metadata index rather than a per-process cache. A staleness verdict cached behind a directory signature was rejected because it would be wrong: a directory's own modification time does not change when a file inside it is replaced in place, so such a cache would report a changed corpus as fresh. At this corpus size the switch does not change how fast a search feels — a warm hybrid query measured 567 ms with the check and 583 ms without it, inside the machine's run-to-run variance — and it is not claimed to.

## 4. Retrieval quality and query latency

Measured through `ResearchService.search`, the engine the workspace and the command line both call, on the current generation (13,158 chunks, 55 documents): 32 judged queries over 19 target passages, `include_staleness=false`, 160 searches in 107 s (mean 0.604 s), after a discarded 10.44 s warm-up. `scripts/evaluate_retrieval.py` produces these tables and a per-query JSON report; `evaluation/README.md` documents the judged set and its protocol.

### Per-mode latency

| Mode | Mean per search | Mean passages returned |
|---|---|---|
| BM25 | 0.084 s | 10.0 |
| Dense | 0.120 s | 5.3 |
| Hybrid | 0.173 s | 10.0 |
| Hybrid + rerank | 2.295 s | 10.0 |

Dense and hybrid are cheap because the index is the exact scan: an embedded ANN index costs 0.5–0.6 s per query at this corpus size, since every query reopens it.

### Accuracy at `top_k=10`

`succ@k` is the share of queries whose judged passage is inside the first `k`; `doc@k` is the share where at least the right document appears, which separates a ranking miss from a coverage miss; `overlap` is the mean share of a query's content words its judged passage contains.

| Mode | succ@1 | succ@3 | succ@10 | MRR | nDCG@10 | doc@10 | overlap | mean returned |
|---|---|---|---|---|---|---|---|---|
| BM25 | 59.4% | 75.0% | 84.4% | 0.671 | 0.713 | 87.5% | 0.659 | 10.0 |
| Dense | 46.9% | 53.1% | 62.5% | 0.513 | 0.539 | 71.9% | 0.659 | 5.3 |
| Hybrid | 65.6% | 78.1% | 84.4% | 0.714 | 0.745 | 90.6% | 0.659 | 10.0 |
| Hybrid + rerank | **81.2%** | **84.4%** | **87.5%** | **0.836** | **0.846** | **93.8%** | 0.659 | 10.0 |

### Accuracy by query class at `top_k=10`

| Mode / class | n | succ@1 | succ@3 | succ@10 | MRR | doc@10 |
|---|---|---|---|---|---|---|
| BM25 / quote | 11 | 90.9% | 100.0% | 100.0% | 0.939 | 100.0% |
| BM25 / paraphrase | 11 | 36.4% | 45.5% | 54.5% | 0.422 | 63.6% |
| BM25 / entity | 10 | 50.0% | 80.0% | 100.0% | 0.651 | 100.0% |
| Dense / quote | 11 | 63.6% | 63.6% | 81.8% | 0.657 | 90.9% |
| Dense / paraphrase | 11 | 54.5% | 54.5% | 54.5% | 0.545 | 72.7% |
| Dense / entity | 10 | 20.0% | 40.0% | 50.0% | 0.320 | 50.0% |
| Hybrid / quote | 11 | 81.8% | 100.0% | 100.0% | 0.879 | 100.0% |
| Hybrid / paraphrase | 11 | 54.5% | 54.5% | 54.5% | 0.545 | 72.7% |
| Hybrid / entity | 10 | 60.0% | 80.0% | 100.0% | 0.718 | 100.0% |
| Hybrid + rerank / quote | 11 | 100.0% | 100.0% | 100.0% | 1.000 | 100.0% |
| Hybrid + rerank / paraphrase | 11 | 63.6% | 63.6% | 63.6% | 0.636 | 81.8% |
| Hybrid + rerank / entity | 10 | 80.0% | 90.0% | 100.0% | 0.875 | 100.0% |

### Deep pass, hybrid at `top_k=50`

succ@50 90.6%, MRR 0.717, nDCG@50 0.759, doc@50 100.0%. Paraphrase reach rises from 54.5% at 10 to 72.7% at 50.

### What these numbers support

- **Reranking is the largest single quality gain**: first-position success rises from 65.6% to 81.2%, MRR from 0.714 to 0.836, and document-level success from 90.6% to 93.8%, for 2.295 s per query against 0.173 s. It is the only configuration that returns the judged passage first for all 11 quote queries, so it is on by default.
- **Hybrid orders better than BM25 alone** — succ@1 65.6% against 59.4%, MRR 0.714 against 0.671, doc@10 90.6% against 87.5% — while reach at 10 is identical at 84.4%. Choosing BM25 for its speed gives up ordering quality, not coverage.
- **Dense alone is the weakest mode** at 62.5% within 10 results, and returns fewer passages (5.3) because the 0.72 cosine gate rejects most of its candidates. Its worst class is entity queries (50% within 10, MRR 0.320): a proper noun needs its words matched, which is what BM25 is for.
- **Paraphrase is the hardest class for every mode** — 54.5% for BM25, dense, and hybrid, and 63.6% after reranking. Four of the eleven paraphrase queries are missed by every mode at `top_k=10`; two of those return the right document at the wrong rank, and depth recovers class reach to 72.7% at 50. The gap is part depth and part ordering, not the relevance gates.
- **The judged set behaves as designed**: mean query-to-target content-word overlap is 0.976 for quote queries, 0.327 for paraphrase, and 0.676 for entity queries, so the class labels describe what they claim to.

### What these numbers do not establish

- The judgments are known-item and single-annotator. A mode that returns a different passage making the same point is scored as a miss, and no pooled judgment exists, so no true recall figure is claimed.
- They come from one English-primary corpus and one generation, not from a benchmark suite.
- Fusion weights are deliberately untouched: with 32 queries the hybrid lead at rank 1 is about two queries, which cannot separate a real weight effect from noise.

### Which product produced them

A number here is comparable only with another measured by the same engine, because a process answering from a different checkout reads a different reviewed-metadata field set and can refuse the project's own state. `doctor` reports the two directories when they differ, `research-rag update` names the processes still running the pre-update code, and the reviewed-metadata file records the fields its writer understood, so an older process refuses the file by cause instead of naming a field the caller never typed.

### Which version an update compares against

`update` reads the remote's release tags with `git ls-remote --tags`, so a check needs no token and no API key. `DIST_TAG` is the tag name that names the current release rather than a version of its own (`latest`); a remote that publishes it wins over the highest version tag, because a maintainer moves it when a release is superseded. A tag carrying a pre-release or build-metadata suffix is not a release, and two tags claiming one version is a refusal rather than a coin toss. A checkout's position against that release is one of `at_release`, `behind_release`, `ahead_of_release`, `no_release`, and `unreadable_release`, and only `behind_release` is an update to apply: `ahead_of_release` is unreleased work, which is a fact to report and never a reason to change anything.

`research-rag` and the frozen `research-ultra-rag-mcp` read and write the same project, which adds one limit this file does not resolve on its own: a figure measured by one and a figure measured by the other are the same measurement, because both call one unchanged engine, but a figure measured by a third checkout is not comparable until `doctor` reports the two trees as the same. Only `research-rag` changes, so a figure here is measured against an engine the frozen product still matches.

### The reranker model: the default against `jinaai/jina-reranker-v1-turbo-en`

Every accuracy number above was measured with the default cross-encoder. The model is an engine setting, not a search option, so the harness can measure a second one over the same judged queries in one run:

```bash
uv run python scripts/evaluate_retrieval.py --project /mnt/data/my-project \
    --modes hybrid,hybrid+rerank \
    --reranker-model Xenova/ms-marco-MiniLM-L-6-v2 \
    --reranker-model jinaai/jina-reranker-v1-turbo-en
```

Measured on the current generation (14,072 chunks, 59 documents) with 30 of the 32 judged queries at `top_k=10` and no deep pass: 90 searches in 191 s. Two queries are missing because they judge target `t12`, whose source the reviewer excluded on 2026-09-20 as superseded by the Duke reprint in "STUART HALL, SELECTED WRITINGS ON RACE AND DIFFERENCE.pdf"; the current generation no longer holds it, so the run names it (`--skip-targets t12`) and reports 30 evaluated queries rather than quietly resolving fewer targets.

| Reranker | succ@1 | succ@3 | succ@10 | MRR | nDCG@10 | doc@10 | mean s | p50 s | max s |
|---|---|---|---|---|---|---|---|---|---|
| none (hybrid) | 66.7% | 80.0% | 86.7% | 0.728 | 0.762 | 90.0% | 0.19 | 0.19 | 0.51 |
| `Xenova/ms-marco-MiniLM-L-6-v2` (default) | **83.3%** | **86.7%** | **90.0%** | **0.858** | **0.869** | **93.3%** | **2.34** | 2.30 | 3.07 |
| `jinaai/jina-reranker-v1-turbo-en` | 73.3% | 86.7% | 90.0% | 0.794 | 0.821 | 93.3% | 3.47 | 3.26 | 11.69 |

By query class, at rank 1:

| Reranker / class | n | succ@1 | succ@3 | MRR |
|---|---|---|---|---|
| ms-marco / quote | 10 | 100.0% | 100.0% | 1.000 |
| ms-marco / paraphrase | 10 | 70.0% | 70.0% | 0.700 |
| ms-marco / entity | 10 | 80.0% | 100.0% | 0.875 |
| jina turbo / quote | 10 | 90.0% | 90.0% | 0.917 |
| jina turbo / paraphrase | 10 | 50.0% | 70.0% | 0.567 |
| jina turbo / entity | 10 | 80.0% | 100.0% | 0.900 |

- **The default stays.** jina turbo reaches the same depth — succ@10 90.0% and doc@10 93.3% are identical — but it puts the judged passage first 22 times out of 30 against ms-marco's 25, and it costs about half again as much per query (3.47 s against 2.34 s). It is therefore neither the better nor the cheaper default here. Its one gain is entity ordering, where MRR rises to 0.900 from 0.875.
- **The two rerankers disagree about rank on only 4 of 30 queries**, and ms-marco is the better of the two on 3 of them: `q06` (quote, rank 1 against 6), `q16` and `q18` (paraphrase, rank 1 against 3 each). jina turbo wins `q27` (entity, rank 2 against 4). A 30-query set can support "the default is not worse", not a fine-grained model ranking; treat the split as directional.
- **The slower model is the smaller one here**, which is an ONNX-export property rather than a parameter to tune: both models run through the same FastEmbed cross-encoder class with the runtime's default thread count, and the first jina turbo call in the run (model load included) is the 11.69 s maximum in the table.
- **Changing the model is an operator decision, not a per-search one**: `--reranker-model` (or `RESEARCH_ULTRARAG_RERANKER_MODEL`) changes every search this app answers, and `search(rerank_model=...)` changes it for one engine call. Re-measure before trusting either on a different corpus: the ordering above is specific to these queries.

### Reply depth: what the reader's top_k buys

The reranker reorders `min(candidates, rerank_max_candidates, max(top_k * 2, 10))` passages, so the depth a reader asks for is the depth the ranking reaches: at `top_k=8` only sixteen candidates are reranked, and the cap of fifty never binds at any depth measured here. That makes `top_k` the cheapest quality lever this app has.

Measured on the current generation with reranked hybrid, 30 of the 32 judged queries and no deep pass, with the payload size of one search at the same depth:

| `top_k` | succ@1 | succ@3 | succ@k | MRR | nDCG@k | doc@k | answer size |
|---|---|---|---|---|---|---|---|
| 8 | 80.0% | 83.3% | 86.7% | 0.825 | 0.835 | 90.0% | 22,151 B |
| 10 | **83.3%** | 86.7% | 90.0% | **0.858** | 0.869 | 93.3% | 25,808 B |
| 12 | **83.3%** | 86.7% | 90.0% | **0.858** | 0.869 | 93.3% | — |
| 15 | **83.3%** | 86.7% | **93.3%** | **0.861** | **0.877** | **96.7%** | 38,532 B |

- **The plateau starts at ten, and eight is the only depth measured below it.** One query out of thirty separates 8 from the rest, so the difference is directional rather than decisive on this set; it is consistent across every metric, and the mechanism is visible in the window arithmetic above: 8 reranks sixteen candidates, 10 reranks twenty.
- **The default is 10**, the smallest depth that keeps the measured quality, and the depth every published number was taken at. Asking for 15 buys reach (`doc@k` 93.3% to 96.7%), which is worth it when a first answer is thin.
- **The payload grows by roughly 1.6 kB per passage.** Depth is cheap, and a workspace renders it either way.
- **`rerank_max_candidates` and the minimum candidate count are not levers at this corpus size.** Raising the cap to 100 or 150, and the minimum to 50, changed no ranking decision, because the window formula never reaches them. They stay settings because a larger corpus could reach them, not because they moved anything here.
- **`maximum_candidates` bounds the widening a filter triggers, and that is where it acts.** A filter drops passages after the corpus is ranked, so a narrow one cannot fill `top_k` and the lexical window doubles until it can; without a bound that doubling walks the whole corpus, a full ranking per step, and a single-author query over 23,272 chunks spent 59.8 s to answer. The window stops at `maximum_candidates` (200), which answers the same query in 0.3 s, and the response reports the window it reached so a filter that matched nothing inside it cannot read as an empty corpus. The bound costs reach: a document whose passages all rank below the ceiling is not found by a filter that names it, which the earlier full-corpus walk did reach.

## 5. Current limits

- **Retrieval quality is measured, not settled.** The numbers above describe findability of one designated passage per query on one corpus; pooled judgments, a second annotator, and a second corpus are open work in `TODO.md`.
- **Dense cost above this corpus size is extrapolated.** The 200,000-chunk switch to the embedded backend is arithmetic from the measured exact-scan cost, not a measurement at that size.
- **Over-limit chunks are flagged, not split.** Splitting would change chunk identities and reuse behaviour, so the build counts and flags them.
- **Older generations are read as they are.** A schema-1 generation stays BM25-only until it is re-ingested, and a lookup written before the retrieval-verdict column is recomputed on first use rather than regenerated silently.
- **Generation retention is unbounded.** Nothing prunes generations; `status` reports what each occupies and removal is manual.
- **The embedding thread count is left to the runtime.** The measured optimum is machine-specific, and forcing the value measured worse than the default on the reference machine.
- **The embedded backend is unoptimised for corpora far above this one.** One connection per phase and time-boxed upload batches are unbuilt, and both matter only above the 200,000-chunk threshold where that backend is selected at all; nothing here measures that size, so the cost above it is arithmetic rather than a reading.
- **Extraction is English-oriented.** The embedding model, the text-health policy, and the chunker target English-primary prose; other scripts appear as quotations inside it and are marked advisory rather than withheld.
- **`assembly` cost sits inside the machine's noise.** It stayed under 1.7 s in every measured run, so no claim is made about it.

## What a ranking change costs the corpus

The reuse snapshot never depended on the ranking policy. `generation_is_reusable` compares the schema, extraction, cleaning and artifact policy versions, the project id, the chunk size and overlap, and the embedding model — and nothing else — so weights, gates, caps and the reranked window cannot affect what a corpus holds. Measured on the German reference corpus, changing `retrieval.rrf_k` from 60 to 30 and re-ingesting:

| Quantity | Result |
|---|---|
| new generation | yes (`generation_changed: true`) |
| documents reused / rebuilt | 13 / **0** |
| chunks reused / rebuilt | 9,237 / **0** |
| vectors reused / created | 9,237 / **0** |
| phase timings | extraction 9.4 s, chunking 6.0 s, embedding 35.3 s, bm25 2.2 s, assembly 20.6 s |

About two minutes end to end against a full build's seventeen, so a retrieval experiment that reaches a new baseline is cheap to apply.

The staging checkpoint identity does not include the retrieval-policy fingerprint (`INGESTION_IDENTITY_POLICY_VERSION` 3), so editing a ranking value resumes an interrupted build rather than discarding it, and a build in progress survives the edit. Only disposable staging is identified this way: a published generation records its policy in its manifest, and a test pins that a ranking change reuses every chunk and vector.

## The cosine gate, calibrated

`retrieval.dense_minimum_cosine_similarity` withholds a dense candidate that scores below it, before the fusion sees it. Swept it with everything else at the shipped values, against the same generation and the same 30 evaluated queries:

| Gate | dense succ@1 | dense results | hybrid succ@3 | hybrid MRR | fused pool | reranked succ@1 | reranked MRR | reranked doc@k |
|---|---|---|---|---|---|---|---|---|
| 0.0 (off) | 56.7% | 10.00 | 76.7% | 0.706 | 69.8 | 83.3% | 0.857 | 96.7% |
| 0.6 | 56.7% | 9.93 | 76.7% | 0.706 | 67.2 | 83.3% | 0.857 | 96.7% |
| **0.72 (shipped)** | 46.7% | 5.23 | 80.0% | 0.712 | 46.8 | 83.3% | 0.858 | 93.3% |
| 0.8 | 10.0% | 0.77 | 80.0% | 0.689 | 40.1 | 80.0% | 0.825 | 90.0% |

Below 0.72 the gate does almost nothing: 0.6 is indistinguishable from off on every metric and removes 2.6 of 70 candidates a query. At 0.72 it removes 23 candidates a query from the fused pool and the shipped path is unchanged — succ@1 is identical, and it trades one top-3 hit for one document. At 0.8 it stops removing noise and starts removing relevant passages: the paraphrases lose a query, `doc@k` falls from 93.3% to 90.0%, and a dense-only ranking returns 0.77 results a query, so the fused answer has nothing left but its lexical half.

So 0.72 it is: the shipped path cannot tell the values below it apart, and the value above it is where the gate starts costing answers. The default stays, and the measurement bounds the knob rather than crowning a value.

The gate is calibrated for a fused ranking, and a dense-only one wants a much lower value — at 0.72 it returns 5.2 results where 10 were asked for. The agent-facing search tool is hybrid-only, so nothing shipped is in that position, and a dense-only method exposed later would have to not apply this gate. The relative rescue measured below narrows that gap without moving the floor.

## The dense floor's relative rescue, measured

`retrieval.dense_relative_similarity_margin` answers the case the floor alone gets wrong: a query whose whole candidate list sits in a band just under it. A floor decides on score alone, so a one-word query — whose best passage can score below the floor while its neighbours sit a few hundredths behind — loses most of its semantic candidates before fusion sees them. On the reference corpus `waste` had fifty dense candidates spanning 0.681 to 0.742, and the 0.72 floor kept six of them over four sources. The margin admits a below-floor candidate when the query's best candidate cleared the floor and this one is within the margin of it, and admits nothing when nothing cleared the floor, which keeps abstention intact.

Swept it with everything else at the shipped values, against the same generation and the same 30 judged queries (81 indexed sources, 19,400 chunks):

| Mode | Margin | succ@1 | succ@3 | succ@k | MRR | nDCG | doc@k | mean sources | mean results |
|---|---|---|---|---|---|---|---|---|---|
| dense | 0.0 | 26.7% | 40.0% | 53.3% | 0.353 | 0.397 | 60.0% | 1.6 | 4.4 |
| dense | **0.10 (shipped)** | 26.7% | 43.3% | 60.0% | 0.371 | 0.426 | 63.3% | 2.1 | 7.1 |
| dense | 0.20 | 26.7% | 43.3% | 60.0% | 0.371 | 0.426 | 63.3% | 2.1 | 7.3 |
| hybrid | 0.0 | 56.7% | 70.0% | 86.7% | 0.646 | 0.698 | 90.0% | 7.7 | 10.0 |
| hybrid | **0.10 (shipped)** | 63.3% | 73.3% | 86.7% | 0.688 | 0.730 | 90.0% | 7.3 | 10.0 |
| hybrid | 0.20 | 63.3% | 73.3% | 86.7% | 0.688 | 0.730 | 90.0% | 7.4 | 10.0 |
| hybrid + rerank | 0.0 | 80.0% | 83.3% | 86.7% | 0.823 | 0.834 | 90.0% | 7.5 | 10.0 |
| hybrid + rerank | **0.10 (shipped)** | 80.0% | 83.3% | 86.7% | 0.823 | 0.834 | 90.0% | 7.5 | 10.0 |
| hybrid + rerank | 0.20 | 80.0% | 83.3% | 86.7% | 0.823 | 0.834 | 90.0% | 7.5 | 10.0 |

The shipped path — hybrid with reranking, the only thing the tool does — is unchanged at both values: every column is identical to the floor-only run. What moves is where the floor was starving a mode. Dense-only returns 7.1 passages where it returned 4.4 of the 10 asked for, and its succ@k rises from 53.3% to 60.0%; unreranked hybrid gains a query and 0.042 MRR. Ten hundredths is the plateau: 0.20 adds 0.1 of a source and 0.2 of a passage to dense-only and moves nothing else, so the default is the shallower of the two.

On the case that prompted it, `waste` on this corpus: the candidate pool goes from 42 over 7 sources to 73 over 12, with 34 candidates admitted below the floor and none rejected. What still binds the *answer* at the default depth is the reranked window, not the floor: it holds 20 candidates and the reranker's order is concentrated, so a ten-passage answer still comes from six sources. Asked for 25, the same query returns 17 sources, including Gabrys, Kimani, OECD, Crawford, and three Gidwani texts that the floor alone never admitted.

The margin is a runtime setting, like the source-diversity penalty: it enters neither the retrieval-policy fingerprint nor the generation manifest, so generations built before it keep validating. `status` reported `generation_upgrade_required: false` on the generation these runs measured.

Two disclosures accompany it, both full-detail only. `dense_gate` reports the floor, the margin, the query's best cosine similarity, and how many candidates were admitted or rejected below the floor. `rejected_candidate_examples` names up to `retrieval.maximum_withheld_examples` sources per reason, so a thin answer reads as thinned rather than silent.

## The minimum passage length, measured

Chunks never span extraction units — Crawford's *Atlas of AI* yields 3,426 chunks from 3,424 units — so one short unit becomes one short chunk. A book's back matter is made of short units, and an index line is a list of the corpus's own words: `value production chain` returned the five-word entry "value production chain, 70–71" from Crawford's Index, which matches the query and carries nothing to cite. Corpus-wide, 3,229 of 19,400 chunks (16.6%) are under twelve words, 993 of them in Crawford's index and notes alone, and 2.7% of the passages the tool returned across the 30 judged queries were in that class.

`retrieval.minimum_passage_words` drops a candidate whose cleaned text is shorter than the setting, before fusion, in both halves. It ships off, because a corpus of deliberately short passages is a legitimate corpus; the reference project sets 12.

| Mode | Minimum words | succ@1 | succ@3 | succ@k | MRR | nDCG | doc@k | mean sources | mean results |
|---|---|---|---|---|---|---|---|---|---|
| dense | 0 | 26.7% | 43.3% | 60.0% | 0.371 | 0.426 | 63.3% | 2.1 | 7.1 |
| dense | **12** | 26.7% | 43.3% | 63.3% | 0.375 | 0.436 | 66.7% | 2.3 | 7.1 |
| hybrid | 0 | 63.3% | 73.3% | 86.7% | 0.688 | 0.730 | 90.0% | 7.3 | 10.0 |
| hybrid | **12** | 63.3% | 73.3% | 86.7% | 0.688 | 0.730 | 90.0% | 7.3 | 10.0 |
| hybrid + rerank | 0 | 80.0% | 83.3% | 86.7% | 0.823 | 0.834 | 90.0% | 7.5 | 10.0 |
| hybrid + rerank | **12** | 80.0% | 83.3% | 86.7% | 0.823 | 0.834 | 90.0% | 7.4 | 10.0 |

The shipped path is unchanged to four decimals, dense-only improves because a fragment was holding a slot, and no column anywhere falls: no judged target is under 30 words, so nothing the set can measure is lost. `value production chain` with the rule on returns ten passages over nine sources instead of an index entry over one, and the full-detail payload names what it dropped — one BM25 candidate and thirteen dense ones, all from *Atlas of AI*.

Twelve words removes 16.6% of the corpus's chunks; twenty removes 22.9% and also drops 12-to-19-word captions and copyright lines. Like the margin, the setting is runtime and enters neither the fingerprint nor the manifest, so it applies at the next search with no rebuild.

### The same floor in the chunker's own unit

`retrieval.minimum_passage_token_fraction` states the floor as a fraction of the generation's recorded `chunking.size`, counted with the tokenizer that generation was chunked by. It exists because "short" is a unit question: a chunk is built to hold `chunking.size` tokens, so a candidate holding a tenth of that is a fragment whatever its word count, while 12 words is a different quantity on every corpus.

What the reference corpus is made of, counted with that tokenizer (19,400 chunks, 81 documents):

| Measure | Value |
|---|---|
| tokens per chunk | min 1, p10 10, median 148, mean 185.5, p90 384, max 386 |
| content kinds | prose 17,325, figure 1,280, list 754, table 41 |
| chunks under 19 tokens (5% of 384) | 2,929 (15.1%), holding 0.7% of the corpus's tokens |
| chunks under 38 tokens (10%) | 4,583 (23.6%), holding 1.9% |
| chunks under 57 tokens (15%) | 5,885 (30.3%), holding 3.6% |

A quarter of the chunks are fragments and they carry under two percent of the text. The count is taken over the returned text, which is cleaned after chunking: re-chunking the corpus's own units at the recorded configuration and counting with the same tokenizer gives a maximum of exactly 384, so the stored corpus's 386-token maximum is cleaning rather than a counting mismatch, and a token or two cannot matter at a tenth of the chunk size.

Judged-set rows, 30 queries over 18 resolvable targets at `top_k=10`, same generation, `minimum_passage_words=0` so the fraction is the only length rule:

| Mode | Fraction | Floor | succ@1 | succ@3 | succ@k | MRR | nDCG | doc@k | mean sources |
|---|---|---|---|---|---|---|---|---|---|
| dense | 0 | 0 | 26.7% | 43.3% | 60.0% | 0.371 | 0.426 | 63.3% | 2.1 |
| dense | 0.05 | 19 | 26.7% | 43.3% | **63.3%** | **0.375** | **0.436** | **66.7%** | **2.2** |
| dense | **0.10** | 38 | 26.7% | 43.3% | **63.3%** | **0.375** | **0.437** | **66.7%** | **2.3** |
| hybrid | 0 | 0 | 63.3% | 73.3% | 86.7% | 0.688 | 0.730 | 90.0% | 7.3 |
| hybrid | **0.10** | 38 | 63.3% | 73.3% | 86.7% | 0.688 | 0.730 | 90.0% | 7.3 |
| hybrid + rerank | 0 | 0 | 80.0% | 83.3% | 86.7% | 0.823 | 0.834 | 90.0% | 7.5 |
| hybrid + rerank | **0.10** | 38 | 80.0% | 83.3% | 86.7% | 0.823 | 0.834 | 90.0% | 7.5 |
| hybrid + rerank | 0.15 | 57 | 80.0% | 83.3% | 86.7% | 0.823 | 0.834 | 90.0% | 7.4 |
| hybrid + rerank, 12 words | 0.10 | 38 | 80.0% | 83.3% | 86.7% | 0.823 | 0.834 | 90.0% | 7.5 |

Every shipped-path column is the same at every value, at 0.05, 0.10 and 0.15 alike, and the fraction adds nothing to the project's word floor: the last row is the reference project's configuration — 12 words *and* a tenth of the chunk size — and it measures exactly like no floor at all. The word floor and the fraction are not alternatives to weigh against each other on this corpus; the only thing the fraction does here that 12 words did not is state the rule in the unit the chunks are counted in.

Dense-only is where a floor is visible, and there it is a gain rather than a cost: the baseline never returned target `q07`, and every floor at or above 0.05 puts it at rank 8 or 9 once a fragment stops holding the slot. That is one query's movement in thirty, and the only per-query difference anywhere in these six runs.

No column falls at any value. Every judged target is long: the shortest of the 18 is 168 tokens and the next three are 214, 218 and 228, so nothing the set can measure is removed by 38 tokens or even 57.

What the rule does to the two queries that exposed the problem:

| Query | Fraction | Hits | Sources | Shortest passage | Dropped candidates |
|---|---|---|---|---|---|
| `value production chain` | 0 | 10 | 8 | **7 tokens** (the Index entry, at rank 1) | — |
| `value production chain` | 0.05 | 10 | 9 | 43 tokens | 1 lexical, 14 dense |
| `value production chain` | 0.10 | 10 | 9 | 43 tokens | 2 lexical, 16 dense |
| `waste` | 0 | 10 | 6 | 115 tokens | — |
| `waste` | 0.10 | 10 | 6 | 115 tokens | 0 lexical, 2 dense |

The one-word query is the case the word floor was already handling; the fraction reaches the same passages through the corpus's own unit, and `waste` — whose shortest returned passage is 115 tokens — is untouched, as a length rule should leave a query that never had the problem.

Cost and reversibility. The tokenizer loads once per process (0.22 s) and counting costs about 0.05 ms a passage, under the noise of these runs' per-search means (0.49–0.79 s across the six runs here, ordered by machine load rather than by setting). The fraction is runtime like the word floor: it enters neither the retrieval-policy fingerprint nor the generation manifest, and the runs report `generation_upgrade_required: false`, so a project can set it, measure it, and drop it without rebuilding.

What these numbers do not establish. The judged set's shortest target is 168 tokens, so these rows bound the harm to long designated passages and say nothing about a corpus whose evidence is legitimately short. The 38-token floor is not a back-matter filter: it takes 23.4% of the corpus's *prose* chunks (4,057 of 17,325), 56.1% of its lists (423 of 754) and 70.7% of its tables (29 of 41), because a short unit is short whatever it holds. On this corpus that is the intended trade — a five-word index line is exactly as short as a one-line paragraph, and only the query can tell them apart — but a corpus whose answers are table rows, catalogue entries, or bibliography lines would lose evidence the judged set here could never detect. That is why the packaged default is 0 and the value belongs in a project's own configuration.

## The lexical abstention gate, measured

The gate drops a lexical candidate that shares no query token outside the corpus language's function-word set. English takes bm25s's fuller list, which also stops question and do-support words, so `how?`, `why not?`, and `what does it do?` share nothing with any candidate and abstain instead of matching whatever passage contains the word they asked with. A corpus in several languages unions every named language's list, and the fallback words stay stopped even when bm25s is unavailable.

The judged set holds no contentless query, so it cannot measure the new behaviour; `tests/test_service.py::test_contentless_queries_abstain` and the settings tests pin it. What the set measures is the cost to content-bearing questions. Run on the current generation (81 sources, 19,400 chunks, 30 queries, `--skip-targets t12`, `top_k=10`), 150 searches in 101.7 s:

| Mode | succ@1 | succ@3 | succ@k | MRR | nDCG | doc@k | mean returned | mean sources |
|---|---|---|---|---|---|---|---|---|
| bm25 | 60.0% | 76.7% | 83.3% | 0.677 | 0.715 | 90.0% | 10.0 | 4.5 |
| dense | 26.7% | 43.3% | 63.3% | 0.375 | 0.437 | 66.7% | 7.1 | 2.3 |
| hybrid | 63.3% | 73.3% | 86.7% | 0.688 | 0.730 | 90.0% | 10.0 | 7.3 |
| hybrid + rerank | 80.0% | 83.3% | 86.7% | 0.823 | 0.834 | 90.0% | 10.0 | 7.5 |

Every column of every row matches this generation's recorded rows at the same gate and floor configuration — the dense and hybrid rows in "The same floor in the chunker's own unit", and hybrid + rerank in both that table and "The minimum passage length" — so the gate change costs the judged set nothing.

## The chunk size and overlap, measured

`chunking.size` is bounded at 384 tokens by its own setting, and that ceiling is the embedding model's: the packaged model reads 512 tokens, so a chunk has to fit inside that with its contextual header. A 512-token chunk is therefore not expressible, which is why this sweep tests *down* from the shipped size and tests the overlap separately.

Three generations were built from the same 81 sources with the same extraction, the same embedding model, and `chunking.headers = true`, differing only in the knob under test. Extraction and chunking are disk-bound: the two experimental builds ran on this machine's 5400 rpm `/mnt/DATA` and spent 3,262 and 3,269 s extracting and 2,096 and 2,091 s chunking, while the reference generation, whose state root is on the NVMe device, spent 459 s and 75 s. Only the two experimental builds are comparable with each other.

| Generation | size/overlap | chunks | median tokens | mean | chunks below a tenth of the size | share | tokens below |
|---|---|---|---|---|---|---|---|
| shipped | 384/64 | 19,400 | 148 | 185.5 | 4,583 (<38) | 23.6% | 1.9% |
| smaller | 256/64 | 24,693 | 197 | 159.2 | 4,583 (<38) | 18.6% | 1.8% |
| wider overlap | 384/128 | 20,186 | 187.5 | 200.3 | 4,583 (<38) | 22.7% | 1.7% |

**The fragments are the same chunks in all three.** Below 25 tokens, below 38, and below 57, every variant holds 3,488, 4,583 and about 5,885 chunks holding 36,212, 69,841 and about 130,500 tokens — the same counts to within the one or two re-tokenized boundaries that splitting produces. They are extraction units that were already short, and neither the chunk size nor the overlap touches them. What changes is dilution: a smaller chunk size splits long units into 5,293 more chunks, almost all above the floor, so the fragment *share* falls from 23.6% to 18.6% while the fragments themselves stay. **A fragment problem is a length-floor problem, not a chunk-size problem**, which is why `retrieval.minimum_passage_token_fraction` is stated in tokens rather than as a hint to re-chunk.

Judged set, the same 21 queries over 13 targets in all three variants, `top_k=10`, the reference project's own policy of 12 words and a tenth of the chunk size. Six of the 19 targets are out of the comparison: five whose snippet straddles a chunk boundary at 256 tokens and so resolves to two overlapping chunks rather than one, and `t12`, whose source the corpus does not hold.

| Mode | Metric | 384/64 | 256/64 | 384/128 |
|---|---|---|---|---|
| hybrid + rerank | succ@1 | 76.2% | 76.2% | 76.2% |
| | succ@3 | 81.0% | 81.0% | 81.0% |
| | succ@k | **85.7%** | 81.0% | 81.0% |
| | MRR | **0.795** | 0.778 | 0.786 |
| | nDCG | **0.810** | 0.786 | 0.792 |
| hybrid | succ@1 | **66.7%** | 57.1% | 61.9% |
| | MRR | **0.722** | 0.631 | 0.692 |
| bm25 | succ@3 | **76.2%** | 61.9% | 66.7% |
| | MRR | **0.690** | 0.646 | 0.679 |
| dense | succ@1 | 23.8% | **33.3%** | 23.8% |
| | MRR | 0.321 | **0.369** | 0.310 |

Query by query, on the shipped path: at 256 tokens three of the 21 change — `q26` loses its target entirely and `q25` slips one rank, while `q27` improves from rank 5 to rank 1. At 128 overlap one changes: `q27`, which the shipped setting finds at rank 5 and the wider overlap never returns. Every moved query is an entity or a quote query, the kind whose evidence is a name, a phrase, or a term list, and both a shorter chunk and a doubled overlap change where such a passage begins.

So neither alternative is taken:

- **A smaller chunk costs the shipped path** a target at `succ@k` (85.7% to 81.0%), 0.017 MRR and 0.024 nDCG, and costs hybrid 0.091 MRR and BM25 fourteen points of `succ@3`. Its one gain is dense-only `succ@1` (23.8% to 33.3%), the weakest mode and not the default, and its better fragment *share* is dilution rather than removal.
- **A wider overlap costs the shipped path** the same target at `succ@k` plus 0.009 MRR and 0.018 nDCG, and costs BM25 `succ@3` (76.2% to 66.7%) while adding 786 chunks for the same text. It is the smaller loss of the two and buys nothing.
- **The shipped 384/64 wins every column the tool's own path reads**, so the default stays and the sweep's finding is that the knob is already at its best measured setting rather than merely at its ceiling.

What this does not establish. The comparison is small: 21 queries over 13 targets, where one target's rank is worth about four points of `succ@k`, so it bounds a decision rather than crowning a configuration. The entity-and-quote pattern is the explanation to test first on a corpus that disagrees. The three generations are within 27% of each other in size, so nothing here says what happens to a much larger corpus, where the chunk count is also what decides the dense backend.

## The reranked window, calibrated

The window is `max(top_k * retrieval.rerank_window_multiple, retrieval.rerank_window_floor)`, capped by `retrieval.rerank_max_candidates` and the fused candidate count — 20 at the default `top_k` of 10. Swept shallower and deeper with the gate at 0.72:

| Window | reranked succ@1 | succ@3 | succ@k | MRR | nDCG | doc@k | mean s |
|---|---|---|---|---|---|---|---|
| 10 | 80.0% | 83.3% | 86.7% | 0.825 | 0.835 | 90.0% | 1.14 |
| 15 | 80.0% | 83.3% | 86.7% | 0.825 | 0.835 | 90.0% | 1.34 |
| **20 (shipped)** | 83.3% | 86.7% | 90.0% | 0.858 | 0.869 | 93.3% | 1.70 |
| 30 | 83.3% | 86.7% | 90.0% | 0.858 | 0.869 | 96.7% | 2.55 |

Ten and fifteen are the same run: both give up one query, and the same one. It is a paraphrase — "Why is reprocessing materials not a real solution to overconsumption?" — whose target the fusion ranks between 16 and 20: the cross-encoder puts it first once it is inside a 20-candidate window, and neither shallower window ever sees it. Depth 20 buys exactly that band, and the band holds only that. Thirty reproduces the shipped succ@1 and MRR, finds one more target document at a rank past 10, and costs 50% more time.

The cost is linear and easy to carry: 1.14, 1.34, 1.70 and 2.55 seconds per reranked query for windows of 10, 15, 20 and 30, which is about 0.07 s for each candidate the cross-encoder reorders plus about 0.44 s fixed. Ten more candidates cost about 0.7 s.

So the default window stays at 20: it is the shallowest window that reaches the measured plateau, and 15 trades one top-1 hit for 0.36 s. Absolute latency is comparable only within one sweep, and this one is the sweep to read: the same configuration measured 1.70 s per reranked query here against 2.3 s in section 4, and the difference is the machine's state rather than the setting.

## Pseudo-relevance feedback, measured

`retrieval.prf` searches the lexical half once, mines terms from the leading passages, searches again with the terms added, and ranks with the second result. Selection weights each candidate by how many leaders use it times how rare it is across the generation, from a document-frequency table built on the first query that needs one and kept while that generation stays loaded, so the cost stays off the query path and a process with the feature off never reads the corpus for it.

Turned on and off against the same judged set and the same generation:

| Configuration | succ@1 | succ@3 | succ@k | MRR | nDCG | doc@k | mean s |
|---|---|---|---|---|---|---|---|
| `prf=false` (shipped default) | 83.3% | 86.7% | 90.0% | 0.858 | 0.869 | 93.3% | 2.34 |
| `prf=true`, 5 leaders, 8 terms | 83.3% | 86.7% | 90.0% | 0.858 | 0.869 | 93.3% | 2.09 |

No quality metric moves, and the mean is run order rather than the feature: the second run met warm caches, and the expansion can only add a pass.

The terms are the ones the feature was designed to add. A direct search with `RESEARCH_ULTRARAG_RETRIEVAL_PRF=true` returned `prf_terms: [verleugnung, cyberculture, hybridization, object, freud, according, virtual, worlds]`: content words, one of them the corpus's own critical vocabulary. Ranking by leader support alone mines the words every passage shares instead, which is why the rarity weight exists.

So the selection rule is doing its job and the judged set cannot see the difference. A known-item set — one designated passage per query, judged by one annotator — scores a passage that makes the same point as a miss, so it registers a change only when the change moves that exact passage. Widening the judgments is what would measure this feature; the default stays off until then.

## Contextual chunk headers, measured

`chunking.headers` prepends the source title and, where a locator carries one, the section to the text a chunk is *embedded* from. The passage itself is untouched, so returned text and citations stay as they were, and the header lives in a separate `embedding_text` on the chunk. The reference corpus was rebuilt with it on and measured against the same 30 judged queries:

| Configuration | succ@1 | succ@3 | succ@k | MRR | nDCG | doc@k | mean s |
|---|---|---|---|---|---|---|---|
| off | 83.3% | 86.7% | 90.0% | 0.858 | 0.869 | 93.3% | — |
| on | 83.3% | 86.7% | 90.0% | 0.858 | 0.869 | 93.3% | 1.40 |

Both generations hold 14,410 chunks, so they differ in the headers and nothing else. Every metric is identical, and so is every query class: quotes 10 of 10, paraphrases 7 of 10, entities 8 of 10 either way.

Part of the reason is visible in the chunks themselves. This corpus is mostly PDFs, and a PDF locator carries a page rather than a section, so a PDF chunk is headed by its title alone — and the first chunk of a paper usually repeats that title in its own first line, which the header then duplicates. For the rest of a paper the header adds the work's name where the passage had none, which is the case the feature exists for, and the judged set cannot see it: it is known-item, so it registers a change only when the exact designated passage moves, the same limit that made the pseudo-relevance expansion measure neutral.

Latency is not comparable across sessions, and this run had no same-session control: 1.40 s per reranked query here against 1.70 s for the same window in the sweep above, with a different machine state behind both.

The default stays off. A project opts in with `chunking.headers = true`, which is what the reference corpus now does. It is an identity setting, so the next ingestion of that project rebuilds — and a re-ingest without it would silently drop the headers, which is why the choice belongs in the project's config rather than on one command line.

## The source-diversity penalty, measured

`retrieval.source_diversity_penalty` reorders the final `top_k` pick. A candidate's adjusted score is its normalized relevance in the ranking that fusion and the reranker produced — 1.0 at the top, 0.0 at the bottom and for the unreranked tail — charged once for every candidate already taken from the same source, and the best adjusted score wins. It only reorders candidates that were already ranked, so it adds and removes nothing, and a ranking with no score to charge against keeps its own order.

Swept on the current generation (16,778 chunks, 77 indexed sources, 8 reviewed exclusions) over 30 judged queries, hybrid and reranked, `top_k=10`, one run per value, target `t12` skipped:

| Penalty | succ@1 | succ@3 | succ@k | MRR | nDCG | doc@k | mean sources | min | median | max |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.0 | 80.0% | 83.3% | 86.7% | 0.825 | 0.835 | 90.0% | 3.9 | 1 | 3.5 | 8 |
| 0.15 | 80.0% | 83.3% | 86.7% | 0.823 | 0.834 | 90.0% | 6.4 | 2 | 6 | 10 |
| **0.25 (shipped)** | 80.0% | 83.3% | 86.7% | 0.823 | 0.834 | 90.0% | 7.4 | 2 | 8 | 10 |
| 0.4 | 80.0% | 83.3% | 86.7% | 0.823 | 0.834 | 90.0% | 8.0 | 2 | 9 | 10 |
| 0.6 | 80.0% | 83.3% | 86.7% | 0.820 | 0.831 | 93.3% | 8.4 | 2 | 9 | 10 |

The case for the setting is the 0.0 row: four of the thirty ten-passage answers came from a single source, and every penalty value removes that case — no value leaves a one-source answer.

The cost is one query's rank. Query 27's target sits at rank 4 after fusion and reranking, and the reordering moves it to rank 5 at 0.15, 0.25 and 0.4, which is the entire MRR difference (0.825 to 0.823 is one rank-4 becoming a rank-5 in thirty queries), and to rank 9 at 0.6 (0.820). Every success column is identical at every value. At 0.6 a second query, query 14, gains its target *document* at a rank its own passage did not reach, which is what raises doc@k to 93.3%.

The shipped default is 0.25 because 0.15, 0.25 and 0.4 are indistinguishable on every quality column here and differ only in spread, and the middle value leaves both directions open. A stronger charge displaces more of a source's own relevant passages, which a known-item judged set cannot score: it registers that the designated passage moved, never that a run of adjacent passages became less useful. 0.4 is there for a project that would rather have nine sources of ten than eight, and 0.6 is where a quality column first moves.

The reordering costs no measurable time: the greedy pass runs over candidates that were already scored and returns `top_k` of them, so the per-search means of these runs (1.6–2.1 s) track machine state rather than the value, as in the sweeps above.

Section 4's tables were taken without this setting, which is the 0.0 row: the same engine with the reordering off. Absolute numbers are comparable only within one sweep, and these rows are the sweep to read.

## What a running project costs the machine

Measured on the reference machine (AMD Ryzen 7 4800H: 8 physical cores, 16 threads, 15 GB) with both corpora's UIs and servers running.

A reranked search at `top_k=40`, about 50 reranked candidates instead of the default 20, with a cold model load each time:

| `runtime.embedding_threads` | wall | total CPU | average cores busy |
|---|---|---|---|
| 0 (the shipped default) | 19.99 s | 76.7 core-s | 3.8 |
| 8 | 18.59 s | 82.9 core-s | 4.5 |
| 4 | 19.20 s | 69.5 core-s | 3.6 |

Wall clock barely moves. Eight threads buy 7% and spend 8% more CPU than the default; four threads cost 3% and save 16%. A search is mostly model loading, and a default `top_k=10` search reranks 20 candidates, about two seconds of work.

Resident memory across every research-rag process, with two projects' UIs and servers alive: **2,055 MB**, roughly 450–500 MB per project, against 15.3 GB of RAM. A project costs that whether or not it is in use.

`runtime.nice` reaches the whole tree because children inherit it. Measured with the setting at 15: the command line, the vanilla gateway it started, and that gateway's extractor and retriever children all reported 15, while processes started before the setting existed stayed at 0 until they were restarted.

Two facts bound what a long build can do to a machine. An `ingest` call yields between atomic units and returns a checkpointed `in_progress` result when the configured `ingestion.work_budget_seconds` runs out, so cancelling between calls loses nothing. Builds are serialized by the project lock, so two ingests cannot compound.
