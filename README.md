# research-rag

- Search a project's PDFs and EPUBs locally on CPU, without an account or hosted model service.
- Read passages with source locators, review bibliography, and exclude unwanted evidence in the browser.
- Use the same knowledge base from the browser, command line, or an MCP agent.
- Results are evidence, not generated answers or verified quotations.

## Requirements

- Linux, Python 3.11 or 3.12, [uv](https://docs.astral.sh/uv/), and Git.
- Internet access for the initial runtime and model downloads. Cached installations can use `--offline`.

## Install

- Install the command:

```bash
uv tool install git+https://github.com/AhmedKishki/research-rag.git
```

- For development, clone the repository and install its command:

```bash
git clone https://github.com/AhmedKishki/research-rag.git
cd research-rag
uv sync --frozen
uv run research-rag install
```

## Build a project

- Choose a project directory and initialize it:

```bash
research-rag --project-root "/path/to/project" init --name "My project"
```

- Copy PDFs and EPUBs into the project's `sources/` directory, then build the index:

```bash
research-rag --project-root "/path/to/project" ingest
```

- The first build can take time. Repeat `ingest` if it reports an unfinished build.

## Read the evidence

```bash
research-rag --project-root "/path/to/project" search "How is evidence interpreted?" --top-k 5
```

- Search combines lexical and semantic retrieval with reranking.
- Passages are cleaned text (`direct_quote_safe: false`). Check the original at the locator before quoting.

## Open the workspace

```bash
research-rag --project-root "/path/to/project" --start-ui start
```

- Keep the terminal open. Ctrl-C or closing it stops the app.
- The workspace has no authentication beyond its loopback boundary. Use a trusted machine.

## Connect an agent

- Start the app, then copy a client entry from the **MCP** view or print it:

```bash
research-rag --project-root "/path/to/project" doctor --mcp-entry
```

- HTTP clients use the app's `/mcp` endpoint. stdio clients use `research-rag mcp --project-name "My project"`.
  - The bridge names a registered project and does not start its app.

## Updates

- The browser's **Updates** view shows GitHub release notes. **Not now** declines without changing the installation.
- `research-rag update` in a terminal shows the changelog and asks for approval. Installation needs explicit approval. `research-rag update --help` covers unattended updates and offline checks.

## Help and details

- `research-rag --help` or `research-rag help`: the command reference, without a project.
- `research-rag COMMAND --help`: arguments and examples for one command.
- `research-rag help TOPIC`: workflow guidance for `agents`, `filters`, or `settings`.
- No OCR is performed. A PDF with an OCR text layer is indexed as that text. A PDF with no text layer or a password is refused.
- [Features and limits](FEATURES.md), [storage and portability](STORAGE.md), [measurement protocols](MEASUREMENTS.md).

## UltraRAG credit and licensing

- Built on [UltraRAG](https://github.com/OpenBMB/UltraRAG)'s MCP architecture, corpus chunker, and BM25 retriever.
  - UltraRAG credits THUNLP at Tsinghua University, NEUIR at Northeastern University, OpenBMB, AI9stars, and its contributors.
- This is an independent project, not an official UltraRAG release, and is not affiliated with or endorsed by those organizations or contributors.
- This code uses the [Apache License 2.0](LICENSE). Extraction components have AGPL obligations. [NOTICE](NOTICE) records their terms, model licences, and upstream revisions.
