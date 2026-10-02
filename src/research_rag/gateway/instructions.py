SERVER_INSTRUCTIONS = """\
This server exposes the existing UltraRAG MCP components without changing their
behavior. Component names are prefixed with their namespace, for example
corpus_build_text_corpus and retriever_retriever_search.
Namespaces: benchmark, corpus, custom, evaluation, generation, memory, prompt,
reranker, retriever, router, and sayhello.

Upstream credit: UltraRAG is developed by the UltraRAG team and contributors as
a joint project of THUNLP, NEUIR, OpenBMB, and AI9stars. Canonical source:
https://github.com/OpenBMB/UltraRAG. This gateway is independent and unofficial.

For the official UltraRAG Vanilla RAG workflow, act as the MCP client
orchestrator and preserve outputs between these upstream stages:
1. benchmark_get_data -> q_ls and gt_ls
2. retriever_retriever_init
3. retriever_retriever_search(query_list=q_ls) -> ret_psg
4. generation_generation_init
5. request the MCP prompt prompt_qa_rag_boxed(q_ls, ret_psg, template)
6. generation_generate(prompt_ls=rendered prompt message text) -> ans_ls
7. custom_output_extract_from_boxed(ans_ls) -> pred_ls
8. evaluation_evaluate(pred_ls, gt_ls, metrics, save_path)
prompt_qa_rag_boxed is an MCP prompt, not a tool. A client must support MCP
prompts and pass its rendered message text to generation_generate. For an
interactive question, q_ls may come directly from the user and the benchmark
and evaluation stages may be omitted; retrieval, RAG prompt construction, and
generation are the essential RAG stages.

Use the exact input schemas supplied with each tool. Important call order:
- Corpus: build a corpus before chunking it.
- Before corpus ingestion, inspect the selected input and confirm that the user
  intends every supported file in it to become source material.
- Dense retrieval: retriever_init, then retriever_embed and retriever_index when
  building; after an existing index is available, retriever_init then
  retriever_search. This is the retrieval path used by the official Vanilla RAG
  example.
- BM25: retriever_init with backend='bm25', then bm25_index when building. In
  UltraRAG 0.3.0.2, initialize once more to load the newly saved index before
  bm25_search. BM25 may replace the dense retrieval stages in a CPU-oriented
  variant while the prompt and generation stages remain the same.
- Reranking and generation: call the corresponding *_init tool before use.

The inherited build tool generates UltraRAG server metadata; it is not document
ingestion or retrieval indexing. Tools that build, embed, index, evaluate, save
memory, or initialize services may write files or consume substantial resources.
Web/API tools may use the network and credentials. GPU/vLLM tools are not part
of the initial CPU-tested path. Confirm explicit paths and overwrite intent
before write operations, and keep runtime paths under the configured workspace.

Direct retriever calls return passages to the MCP client and do not create a
separate search-results file. This vanilla gateway does not add project-root
enforcement, file-type enforcement, metadata filters, citation locators, a
read-only policy, or any other extension behavior.
"""
