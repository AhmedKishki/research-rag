# Roadmap

- Nothing here is scheduled. Each idea needs a decision before work starts.
- Work outside the envelope in `MEASUREMENTS.md` is a different product, not a roadmap step.

## Upstream reuse

- Re-evaluate UltraRAG's dense index backends (`faiss`, `qdrant`, `milvus`) and its reranking components when an upstream release does all of the following:
  - Returns identifiers and scores under query-time metadata filters.
  - Offers a reranker that is CPU-only, offline, revision-pinned, and free of services and credentials.
- Adoption would be a component swap behind the `DenseBackend` boundary plus a new manifest backend name.
  - The source and retrieval integration tests in `AGENTS.md` must accept it before any generation selects it.

## Answers

- `answer` is white-box only, through DeepSeek. Each idea below widens that contract.
- A local model keeps the corpus on the machine but needs a GPU and a serving surface.
- AI-authored links: a per-request switch that lets the model fill a gap as a marked `⟦AI-AUTHORED⟧` link.
- A black-box mode: free prose that cites passages. It needs its own citation check and its own refusal of a thin result set.
- Saved answers: each answer and its derivation under the project, so it can be reopened and rendered again. A new on-disk layout.

## Citations and quotation

- Character offsets inside extraction units, so a hit points at a span.
- The printed page label distinguished from the physical page wherever a document carries both.
- An exact-quote verification tool.
- Citation export in common bibliographic styles, without inventing metadata.

## Scale and operations

- A persistent document-metadata index, for tens of thousands of sources.
- Category and keyword filtering inside the embedded dense index, which matters only above the threshold that selects that backend.
- Incremental dense-index construction for large collections. The embedded backend rebuilds its index for every changed generation.

## Other

- Backup profiles that exclude originals, for readers who store their PDFs elsewhere.
- Cross-project search that keeps each project's boundary explicit.
