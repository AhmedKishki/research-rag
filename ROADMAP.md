# Roadmap

Product ideas that are deliberately not being built yet. Nothing here is scheduled, and each would need a decision before it was started. Work that is already in scope is in `TODO.md`, current behaviour is in `README.md`, and current facts and limits are in `MEASUREMENTS.md`.

The app targets English-primary born-digital PDF and EPUB sources of 5,000–50,000 chunks on CPU. Anything outside that envelope — other-language corpora, OCR'd or scanned material, handwriting, formula-heavy documents — is a different product rather than a roadmap step, so it is not listed here as scheduled work. The one such gap that keeps being reported is recorded at the end of this file, so the decision is visible rather than forgotten.

## Upstream reuse

- Re-evaluate adopting UltraRAG's dense index backends (`retriever_init(index_backend="faiss"|"qdrant"|"milvus")`) and its reranking components once an upstream release returns identifiers and scores under query-time metadata filters, and offers a reranker that is CPU-only, offline, revision-pinned, and free of any service or credential.
- If those criteria are met, the work is a component swap behind the `DenseBackend` boundary plus a new manifest backend name, and the source and retrieval integration coverage in `AGENTS.md` must accept it before any generation selects it.

## Citations and quotation

- Character offsets inside extraction units, so a hit can point at a span instead of a whole unit.
- The printed page label distinguished from the physical page, wherever a document carries both.
- An exact-quote verification tool: the missing piece between cleaned semantic text and text that is safe to quote.
- Citation export in common bibliographic styles, without inventing any metadata.

## Scale and operations

- A persistent document-metadata index, which a corpus of tens of thousands of sources would need before per-process caches of the document map or of the staleness verdict are worth their invalidation risk.
- Pushing category and keyword filtering into the embedded dense index, which matters only above the 200,000-chunk threshold where that backend is selected.
- Incremental dense-index construction for large collections, where the embedded backend currently rebuilds its index for every changed generation.

## Other

- Optional backup profiles that exclude originals, for readers who already store their PDFs elsewhere.
- Optional cross-project search that keeps each project's boundary explicit rather than merging indexes into one.

## Outside the documented envelope

- **OCR before ingestion, so a scanned source could be indexed.** Scanned material sits outside the workload this app is built and measured for — English-primary born-digital PDF and EPUB on CPU — so this is a change of product envelope rather than a step. It would need its own accuracy expectations, its own tests, and an answer to whether an OCR'd source can share a project with a digital one, since the two have different evidence quality.
