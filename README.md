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
- If a PDF or EPUB cannot be extracted, a validated generation containing the remaining sources is retained as **partial**, not selected. Review the omitted sources in the result or generation list. Repair them and run `ingest` again, or select the partial generation with `research-rag --project-root "/path/to/project" generations --use GENERATION_ID`.

## Read the evidence

```bash
research-rag --project-root "/path/to/project" search "How is evidence interpreted?" --top-k 5
```

- Search combines lexical and semantic retrieval with reranking.
- Passages are cleaned text, not a transcript. Check the original at the locator before quoting.

## Open the workspace

```bash
research-rag --project-root "/path/to/project" --start-ui start
```

- Omit `--project-root` and `--project` to choose a registered project in the terminal. Use Up/Down and Enter; Escape or Ctrl-C cancels. This also applies to a bare `research-rag` invocation.
- Keep the terminal open. Ctrl-C or closing it stops the app.
- The workspace defaults to local-only access.

### Use a phone on home Wi-Fi

```bash
research-rag --start-ui start --lan
```

- Open **Remote Access** on the laptop website. Scan its QR code with your phone or open the displayed URL. QR codes are generated locally, not by an external service.
- The existing website remains at `/`. The phone workspace is available alongside it at `/next/`; both use the same backend and project.
- LAN mode uses HTTP without a login. Devices able to reach the port on the private network can read this project's sources and perform enabled workspace actions. Traffic is unencrypted. Use only a trusted home network; never forward this port to the internet.
- LAN access supports the laptop's detected private IPv4 addresses. Allow the displayed port through the laptop firewall for your home network. Guest Wi-Fi or client isolation can prevent access.
- Keep the laptop awake and its serving terminal open. If the laptop's network address changes, restart with `--lan` and scan the new code.
- MCP, CLI control, installation management, and the local project/client registry remain local-only. Start without `--lan` to disable phone access.

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
- Extraction is text only. A PDF with a text layer is indexed as that text; a PDF with no text layer or a password is refused.
  - Scanned PDFs need OCR performed outside this app. Add the recognised copy to the sources directory and run `ingest`.
  - PDF cleaning repairs passages locally and leaves visible gaps where damaged text cannot be recovered. `ingestion.maximum_unclean_percent` caps substantive text loss per document, excluding confirmed furniture and non-evidence. Over-budget or unreadable PDFs are omitted from a retained partial generation, not used to fail otherwise readable sources.
- [Features and limits](FEATURES.md), [storage and portability](STORAGE.md), [measurement protocols](MEASUREMENTS.md).

## UltraRAG credit and licensing

- Built on [UltraRAG](https://github.com/OpenBMB/UltraRAG)'s MCP architecture, corpus chunker, and BM25 retriever.
  - UltraRAG credits THUNLP at Tsinghua University, NEUIR at Northeastern University, OpenBMB, AI9stars, and its contributors.
- This is an independent project, not an official UltraRAG release, and is not affiliated with or endorsed by those organizations or contributors.
- This code uses the [Apache License 2.0](LICENSE). Extraction components have AGPL obligations. [NOTICE](NOTICE) records their terms, model licences, and upstream revisions.
