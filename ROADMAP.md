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

- Add a generation stage, through a local model or a hosted API.
  - Today `search` returns passages, sources, and locators, and the reader writes the answer.
  - This is a change of contract, not a missing feature.
- Requirements:
  - A provider behind a setting, because the CPU-only, offline, credential-free rule does not survive a network call or a GPU.
  - An explicit statement of what leaves the machine: the query and the passages it retrieved.
  - A citation contract, so a generated sentence names the passages it rests on.
- A local model keeps retrieval offline but needs a GPU and a serving surface.
- Constraints:
  - The evidence stays in the payload with its locators.
  - A generation is labelled as one.
  - A request for an answer without evidence is refused, not answered from the model's memory.
- Decisions first:
  - Whether the app keeps returning evidence only and offers generation as a separate operation.
  - What an answer looks like when the reranker found nothing. A model writing from a thin result set writes from its own knowledge.

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
