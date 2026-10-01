## Purpose

`research-rag` reviews evidence from a project's own PDF and EPUB corpus. You point it at a directory of original documents, build a searchable generation from them, and read every passage beside its source, its locator, and the bibliographic metadata you reviewed. You stay the one who concludes anything: the app retrieves cleaned semantic evidence and never writes an answer, and the untouched original file is the only authority for a quotation.

Everything runs on your machine, on one CPU, with no service to sign up for.

## What the app provides

**One app, one port, three ways in.** The app is a server. When it is up you have a browser workspace, an agent surface, and a command line over the same project, the same project lock, and the same search index. When it is down you have none of them. That is the whole state model: one process owns the project, and everything else is a front end to it.

- **A browser workspace.** Search your corpus, read results with their page numbers and citations, open the original PDF or EPUB beside a passage, and record reviewed metadata or an exclusion for each source.
- **An agent surface.** Seven operations and one resource over MCP, served at `/mcp` on the same port as the workspace. An agent's answer is the same payload the workspace renders, projected down to the fields an agent acts on, which is about half the size.
- **One command line.** `research-rag search` and `research-rag status` reach the same running app, so a terminal answer and a workspace answer cannot disagree.
- **Many projects, one install.** Each project is a directory with its own corpus, its own app process, and its own port; `init` records them, `research-rag projects` lists them, and `--project <name>` names one instead of repeating its path.
- **Clients you can see and drop, from either place.** The workspace's status view lists the agents attached to the app and can end one; `research-rag clients` and `research-rag disconnect` do the same from a terminal, over the same registry, so a drop in the browser is a drop the command line can see. A client that cannot open a socket gets the same surface through `research-rag mcp`, which speaks stdio and proxies to the running app.
- **Immutable generations.** A rebuild writes a new generation and switches to it only when every index is complete, so a failed build leaves the previous generation searchable.
- **Reviewed metadata as first-class state.** Categories, projects, keywords, language, title, and authors are editable plain JSON, applied at read time with no rebuild.
- **Honest provenance.** Every passage carries its source, its locator, and whether its text is cleaned semantic content rather than a transcript.
- **Explicit exclusions.** Exclude a reviewed source from every retrieval surface immediately, reversibly, without touching the file.

## Install once

```bash
git clone https://github.com/AhmedKishki/research-rag.git
cd research-rag
uv sync
```

`uv sync` installs Python, the pinned dependencies, and one console command:

```text
research-rag
```

To install it as a package on a machine with no checkout — which is how it is meant to run once you have a project:

```bash
uv tool install research-rag
# or, into a project-local environment:
uv pip install research-rag
```

The first ingestion downloads the pinned UltraRAG runtime, the embedding model, and the reranker into caches on your machine. After that, `--offline` works.

## Settings

Every tunable lives in one registry, and every default value lives in the packaged `default.toml`. Read the merged settings, with the layer each value came from:

```bash
research-rag --project-root /path/to/project config
```

A layer names only the keys it changes, in this order, each one overriding the one before it:

| Layer | Where |
|---|---|
| packaged default | inside the installed package |
| per-user file | `~/.config/research-ultra-rag-mcp/config.toml` |
| extra file | `--config PATH` |
| project file | `<project>/.research-rag/config.toml` |
| environment | `RESEARCH_ULTRARAG_*` variables |
| command line | `--set KEY=VALUE`, repeatable |

```toml
# <project>/.research-rag/config.toml — only the keys you want to change.
[retrieval]
source_diversity_penalty = 0.15

[dense]
embedding_model = "BAAI/bge-small-en-v1.5"
```

```bash
# One run, without editing any file.
research-rag --project-root /path/to/project \
  --set retrieval.source_diversity_penalty=0.2 search "does the state own the machine"

# A project that always uses a different reranker and a tighter dense gate.
research-rag --project-root /path/to/project \
  --set reranker_model=Xenova/ms-marco-MiniLM-L-12-v2 \
  --set retrieval.dense_relative_similarity_margin=0.12 \
  search "fetishism of technology"
```

An unknown key, an out-of-range value, or a value of the wrong type is refused in every layer, and `config` names the layer a value came from, so a setting that is not taking effect is visible rather than mysterious.

### Working in another language

The corpus language selects BM25's stopword list and is part of the ranking policy, so changing it makes the next build produce a new generation.

```toml
[language]
corpus = "french"
```

A language the embedding model does not cover is reported rather than silently embedded. See `research_rag/embeddings.py` for each model's covered languages and required prefixes.

## Update an existing installation

```bash
scripts/update.sh                      # pull, sync, and report
scripts/update.sh --check              # report only; change nothing
scripts/update.sh --check --offline    # report without touching the network
scripts/update.sh /path/to/project     # also restart that project's workspace
```

The script reports the declared version, the installed version, how far the checkout is behind, and whether an update would change the environment. A workspace is a process, so an update never reaches a running one: stop it and start it again, and `status.version.restart_required` then reads `false`.

## Create an isolated research project

```bash
# A project that does not exist yet.
research-rag --project-root /path/to/new-project init --name "Fetishism of Technology"

# A directory you already work in: only .research-rag is added, nothing moves.
cd /path/to/existing-project
research-rag init --name "Fetishism of Technology"
```

`init` records the project identity, generates the workspace launcher under `.research-rag/bin/`, and links it into the project root as `open-research-rag-ui.sh`. It never overwrites an existing file or symlink, and `status.ui_launcher` reports what it found. Delete the symlink to opt out.

`init` also records the project in this installation's project register, so one command can serve many projects and name them:

```bash
research-rag projects
research-rag --project "Fetishism of Technology" status
```

The register is a pointer file, `projects.json` in the account settings directory, holding each project's id, name, and root. It carries no corpus and no state, so deleting it costs nothing but the names, and `init` rebuilds it. Run `init` again on a project that moved and the record follows it there.

## First use

```bash
research-rag --project-root /path/to/project init --name "My project"
cp ~/Downloads/*.pdf ~/Downloads/*.epub /path/to/project/sources/
research-rag --project-root /path/to/project ingest
research-rag --project-root /path/to/project ui
```

`ingest` extracts every included PDF and EPUB, chunks the text, embeds it, and builds a BM25 and a dense index. The current generation stays active the whole time. `ui` brings the app up and opens the workspace in a browser, which is the same as `start --open`.

To give an agent the same corpus:

```bash
# The app is already up, so this reaches it.
research-rag --project-root /path/to/project clients
```

## Connect an agent

The app serves MCP at `/mcp` on its own port, so a client that can open a socket needs nothing but the URL:

```json
{
  "mcpServers": {
    "research-rag": { "url": "http://127.0.0.1:5051/mcp" }
  }
}
```

A client that only speaks stdio uses the bridge, which makes sure the app is up and then proxies to it:

```json
{
  "mcpServers": {
    "research-rag": { "command": "research-rag", "args": ["mcp"] }
  }
}
```

Set `RESEARCH_ULTRARAG_CLIENT_NAME` to something that names the agent, so the app's client list can tell them apart:

```json
{
  "mcpServers": {
    "research-rag": {
      "command": "research-rag",
      "args": ["mcp", "--project-root", "/path/to/project"],
      "env": { "RESEARCH_ULTRARAG_CLIENT_NAME": "claude" }
    }
  }
}
```

### Print the entry for a project

A port is chosen at start and recorded, so a hard-coded URL above goes stale the first time the port moves. Let the project print the entry for the machine it runs on:

```bash
research-rag --project-root /path/to/project doctor --mcp-entry
```

That prints both shapes, so you can copy whichever your client wants. To check an entry that is already in place without editing it:

```bash
research-rag --project-root /path/to/project doctor --check-entry ~/.config/kilo/kilo.jsonc
```

The output depends on the client's own configuration format, so `doctor` reports which shape it recognises and what it expects. Two ready-to-copy templates are in this repository: `mcp_settings.example.json` for a client that uses an `mcpServers` object, and `kilo-mcp.example.jsonc` for one that uses a Kilo-style `mcp` object. Replace the two absolute paths in either one and the entry is complete.

What an agent gets is the same service the workspace uses, with a projected answer: `status`, `ingest`, `search`, `find_source`, `get_passage`, `set_source_inclusion`, `set_source_metadata`, and the `research://status` resource. `status` also reports where the workspace is and how many agents are attached.

An agent's tools answer one question at a time. `status` is a verdict — `ready`, `stale`, and the `requires` list naming the calls that close the gap — and `find_source` looks up one work by filename, title, or author instead of listing the corpus. The command line prints the same lean `status` answer, and `status --verbose` prints the complete payload; `research-rag sources` and the workspace are where a person reads the inventory.

To see who is attached, and to end one, from the browser or from a terminal:

```bash
research-rag --project-root /path/to/project clients
research-rag --project-root /path/to/project disconnect SESSION_ID
```

In the workspace, the same list is under **Knowledge base status → Attached clients**, with a **Disconnect** action beside each attached agent. Both read the same registry, so a session ended in the browser is gone from the command line too.

## Read the answer, then open the original

A search answers with its passages and the `generation_id` they came from, and with a field only when it has news: `stale` when the corpus moved on, `rerank_fallback` when the reranker did not run, and `generation_upgrade_required` when this generation cannot serve the request. An absent field is the ordinary case, not missing data.

- **`text`** is cleaned semantic text for comprehension and paraphrase. It is never a transcript.
- **`source_relative_path`** and **`locator`** say where to open the original. The locator is the position alone: a page, carrying `page_label` only where the printed label differs from the physical page, or a section for an EPUB.
- **`citation`** is the reference to attach to a claim.
- **`direct_quote_safe`** is `false` on every passage the app builds.

For exact wording, open the original at that locator and quote from it. `direct_quote_safe` is the machine-readable form of the same rule, and it is the only place the rule appears per passage.

### How retrieval is chosen

Retrieval is not a choice you make, and that is deliberate.

- Every search is hybrid with CPU cross-encoder reranking, always. BM25 (weighted `1.25`) and dense (`1.0`) ranks are fused, candidates are gated, and the reranker reorders at most 50 of them. On the judged set in `MEASUREMENTS.md` this is the best of the measured modes, so there is no mode to choose.
- It is the slow path on purpose: about 2.3 s per warm query against 0.17 s unranked, and it may download a second model on first use. `MEASUREMENTS.md` compares the supported rerankers. When a response carries `rerank_fallback`, the model could not be loaded and the order you received is the plain unranked candidate order.
- BM25 candidates need at least one query token outside the corpus language's function words, so a query carrying no topic abstains instead of matching on a function word. Dense candidates need cosine `>= 0.72`, with a narrow rescue for a candidate just below the floor when another candidate cleared it.
- A thin answer is a reason to ask again, not a conclusion. The reranker reorders about twice the number of passages you ask for, so a larger `top_k` deepens the ranking as well as the answer, and the same question in different words is a different search.
- The fusion weights, relevance gates, candidate caps, chunk size, and reranker model are operator settings, not per-call ones. They come from your settings file or from `--set`.

`research-rag search` also carries `--method {bm25,dense,hybrid}` and `--no-rerank`, which exist to reproduce a row of `MEASUREMENTS.md` on demand. The workspace has no equivalent control, and neither does any answer: those flags are a measurement escape hatch on one command, not a retrieval mode a reader chooses.

### Finding and referring to sources

```bash
research-rag --project-root /path/to/project sources
research-rag --project-root /path/to/project passage CHUNK_ID --context-chunks 2
research-rag --project-root /path/to/project metadata "evidence.pdf" \
  --title "Reviewed Marsh Evidence" --author "Field Researcher" \
  --category corrected --year 2025
research-rag --project-root /path/to/project exclude "evidence.pdf" --reason "Reviewed duplicate"
research-rag --project-root /path/to/project include "evidence.pdf"
```

`sources` is the inventory, including `discovered_sources` for files that are not indexed yet. It is also what registers stable source IDs in the project's portable catalog, so an addressable handle survives the original file disappearing.

A metadata edit rewrites only that source's entry, so a hand edit to another entry in the same file survives. Excluding a source removes it from every retrieval surface immediately, without touching the file; the next ingestion omits it from new indexes.

## Research workflow

### What a search can filter on

Six layers are available, all read from the reviewed metadata at query time:

```bash
research-rag --project-root /path/to/project search "commodity fetishism" \
  --category "Commodity fetishism" --category marxism \
  --keyword "fetishism of technology" \
  --author Harvey --title "Fetish of Technology" \
  --top-k 15
```

Categories, projects, languages, authors, and titles match any of their values; keywords match all of theirs. Authors and titles match as case-insensitive substrings, because a name is a phrase rather than a controlled tag.

An empty result with a filter applied means no source in the corpus matches the filter, and the answer names the filters it applied rather than reporting a silent corpus. A filtered-out answer is not a broken index.

### Freshness with every search

Every search re-compares the source directory with the generation and reports `stale`. A stale status names what changed: how many sources were added or modified, which are gone from the directory, and whether reviews or exclusions moved. The previous generation remains searchable throughout. Ask whether to re-ingest; a rebuild reuses compatible work rather than starting over.

A reviewed metadata change is not staleness. It is already effective without a rebuild, and the complete status payload reports `metadata_overlay_active` for it.

### What gets excluded, and what does not

Only regular `.pdf` and `.epub` files beneath the configured source directory are indexed. Markdown, symlinks, and everything outside that directory are ignored. Nothing in the app edits an original.

Corrupt extraction units are omitted whole rather than indexed as garbage, and the count is reported. Script mixing is never a reason to withhold: a foreign-language quotation inside an English source is evidence, and its locator tells you where it is.

## Use the terminal

`research-rag help` prints every command grouped by the work, and `research-rag help TOPIC` explains a subject. Neither needs a project, so both work before you have one:

```bash
research-rag help              # every command, grouped by what you are doing
research-rag help filters      # the six filter layers and the retrieval switches
research-rag help settings     # the four setting layers and what a change costs
research-rag help agents       # the MCP client entry, and what an agent receives
research-rag help search       # one command's own options
```

A subject page is for something no single command can answer; naming a command prints that command's own usage, so the two are the same document.

```bash
# Create and build.
research-rag --project-root /path/to/project init --name "My project"
research-rag --project-root /path/to/project ingest

# Bring the app up, and look at it.
research-rag --project-root /path/to/project start
research-rag --project-root /path/to/project clients

# Read.
research-rag --project-root /path/to/project status
research-rag --project-root /path/to/project search "cobalt heron amber marsh" --top-k 5
research-rag --project-root /path/to/project sources
research-rag --project-root /path/to/project passage CHUNK_ID

# Review, which applies to the current generation without rebuilding it.
research-rag --project-root /path/to/project metadata "evidence.pdf" --year 2025
research-rag --project-root /path/to/project exclude "evidence.pdf" --reason "Duplicate"

# Browse it, and read the settings that apply.
research-rag --project-root /path/to/project ui
research-rag --project-root /path/to/project config

# Stop it, and every process of this app serving this project.
research-rag --project-root /path/to/project stop --servers

# Find out what is wrong with the installation.
research-rag --project-root /path/to/project doctor
```

### Configure a client

```bash
research-rag --project-root /path/to/project doctor --mcp-entry
research-rag --project-root /path/to/project doctor --check-entry ~/my-client.json
```

`--mcp-entry` prints two entries: the URL entry, which points at the app's own port and is what a client that can open a socket should use, and the stdio entry, which runs the bridge beside the running interpreter and is what one that cannot should use. `--check-entry` reads a client file you already have and reports what is wrong with it — an executable that is not a file, a project root that is not this project, a URL that is not loopback or is not the agent surface, a timeout shorter than an ingestion, or two entries for one project. Neither writes anything.

### Ask the doctor

```bash
research-rag --project-root /path/to/project doctor
```

`doctor` prints one line per dependency: ok, warn, blocked, or unchecked. It writes nothing, so it is always safe to run.

```
blocked  ultrarag_runtime      the managed runtime tree does not match the pinned snapshot
                                 repair: research-rag doctor --repair-runtime
warn     code_currency         two checkouts of this package are installed
ok       dense_backend         exact scan, 58 searchable documents
```

The two operations that reach the network are flags, and neither implies the other:

```bash
# Download the pinned models into the shared cache, which --offline then requires.
research-rag --project-root /path/to/project doctor --prefetch-models

# Move a mismatched runtime aside, install the pinned one, and validate it.
research-rag --project-root /path/to/project doctor --repair-runtime
```

A repair that discards evidence moves it aside rather than deleting it.

## Use the workspace

```bash
research-rag --project-root /path/to/project ui            # start and print the URL
research-rag --project-root /path/to/project ui --open     # and open a browser
research-rag --project-root /path/to/project ui --stop     # stop it and what it started
./open-research-rag-ui.sh                                 # the generated launcher
./open-research-rag-ui.sh --port 5100
```

The generated launcher claims the first free loopback port at or above the one it was generated with and records the port it chose, so two projects never serve from the same port. It resolves this app's console script beside the running interpreter rather than trusting `PATH`, passes the project's own root and runtime root explicitly, and stops the whole process group on `--stop`.

`research-rag serve` is the same workspace in the foreground on a fixed port. It is what the launcher runs; you only need it when you want to supervise the process yourself.

The workspace binds loopback only. It has no authentication, which is correct for an address no other machine can reach.

The retrieval controls the measurements already answered are not in the workspace: there is no retrieval mode, no reranking switch, and no chunk tuning, because every search is hybrid and reranked. Metadata, source selection, category partitions, and project metadata are all there.

## Where project data is stored

`STORAGE.md` is the storage contract: every file, every field, what is portable, what is rebuilt, and how a hand edit behaves. Two facts matter most here:

- `sources/` is the authority for exact quotation, and nothing in the app edits it.
- The only state shared between projects is the model cache at `~/.cache/research-ultra-rag-mcp/models/` and the project register at `~/.config/research-ultra-rag-mcp/projects.json`, which names projects and nothing else. Document text, embeddings, indexes, logs, and query state never cross project roots.

The user settings directory is `~/.config/research-ultra-rag-mcp/`. Both names are inherited from the frozen `research-ultra-rag-mcp`, which is still installed on the machines that carry it and resolves the same paths; renaming them would cost every such user their settings and about 150 MB of model downloads. `tests/test_data_roots.py` states why.

## Move or back up a project

A project is self-contained. Copy the directory — `sources/` plus `.research-rag/` — and the copy is a complete project on another disk or machine. `runtime/` is disposable and is rebuilt on the next ingestion.

Keep:

- `sources/` — the originals, and the only authority for a quotation;
- `.research-rag/project.json`, `source-catalog.json`, `source-metadata.json`, `source-exclusions.json` — the project identity and your reviewed decisions;
- `.research-rag/runtime/` — optional; copying it keeps the project searchable without a rebuild.

Continue in one place at a time. Two processes pointed at the same project root, or at a copy that shares a relocated runtime root, are refused by the project lock rather than silently interleaved.

## One project, one app

A project has at most one app. The command line, the browser workspace, and an agent's tools are three front ends to that one process, so all three read one index, one review state, and one set of models.

- `research-rag start` and `research-rag ui` bring it up; `ui` also opens a browser. The workspace is served by the same process, so starting a UI from a terminal and then driving the terminal is one instance, not two.
- A second app for the same project is refused rather than started alongside. The port is claimed by binding it before the server exists, so two apps racing for one port cannot both believe they won, and the failure names the port and the fix.
- Two projects need two apps and two ports. That is expected: the launcher claims the first free port at or above the one it was generated with, and records what it chose, so no two projects collide.
- Any front end can bring the app up. A command that needs the app and finds none starts it; the stdio bridge does the same, so an MCP client is handed a running project rather than an error.

## Operations

| Command | What it answers |
|---|---|
| `projects` | every project this installation registered, and whether an app is serving it |
| `init` | the project identity, the app launcher, and the project's place in the register |
| `status` | the readiness answer, or the complete payload with `--verbose` |
| `ingest` | the current generation, a checkpointed build's progress, or a new complete generation |
| `search` | hybrid reranked evidence, with filters applied from reviewed metadata |
| `sources` | the corpus inventory, including files not indexed yet |
| `passage` | one passage and its neighbours |
| `include` / `exclude` | reversible retrieval decisions, enforced immediately |
| `metadata` | one source's reviewed bibliographic override |
| `config` | every effective setting and the layer it came from |
| `doctor` | one line per dependency, with the command that fixes it |
| `start` / `ui` | the app, started; `ui` also opens a browser |
| `clients` / `disconnect` | the agents attached to the app, and ending one — the same list the workspace shows |
| `mcp` | the agent surface on stdio, proxied to the running app |
| `serve` | the app in the foreground, which is what the launcher runs |
| `stop` | the app, and optionally any process of this app still building |
| `help` | every command grouped by the work, or one page of it; needs no project |

`status` is the exception a person reads most, so it prints the same lean verdict an agent gets and takes `--verbose` for the complete payload; every other command prints the complete payload. An agent's tool answer is projected to the fields an agent acts on, and the mode is the `runtime.tool_detail` setting.

Every command takes `--project <name-or-id>` in place of `--project-root <path>`, so one shell can work on several projects without repeating paths. The two together are refused rather than resolved by precedence.

## How it works under the hood

```
MCP client ── HTTP /mcp ─┐
                         │
stdio client ── bridge ──┤
                         ├── research-rag (one process, one port, one lock)
Local browser ── HTTP ───┤      │
                         │      ├── ResearchService
Terminal ── control ─────┘      ├── shared workspace
                                ├── agent tools and resources
                                ├── control API
                                ├── client registry
                                │
                                ├── project and source policy
                                ├── PDF page extraction
                                ├── EPUB section extraction
                                ├── metadata and provenance
                                ├── immutable generations
                                ├── FastEmbed CPU embeddings
                                ├── project-local dense index
                                ├── reciprocal-rank fusion
                                └── CPU cross-encoder reranking
                                    └── stdio MCP ──> vanilla-ultra-rag-mcp
                                                       ├── UltraRAG chunker
                                                       └── UltraRAG BM25
```

The workspace and the agent surface are routes on the same application, and the command line reaches it over the control API on the same port, so one capability has one implementation and one project lock. The vanilla gateway is an implementation dependency below the app, not a second surface.

Retrieval fuses a BM25 rank from UltraRAG with a cosine rank from a project-local dense index, reranks with a CPU cross-encoder, and reports what every gate rejected. The vanilla gateway is only asked for BM25.

## Limitations and troubleshooting

- **The app is a process, not a daemon.** It does not survive an update; stop it and start it again. An agent attached to a stopped app has no session, and `clients` says so rather than listing a stale one.
- **The first build is slow.** It downloads the pinned UltraRAG runtime and two models, then extracts and embeds every source. `status.ingestion_progress` reports progress and a cancelled build resumes where it stopped.
- **`--offline` fails if anything is uncached.** Run `doctor --prefetch-models` first.
- **A build reports "one build at a time".** Another process holds the project lock. `status` names the resident build's phase.
- **Two checkouts of this package.** `doctor` reports `code_currency` as a warning. Two copies of one version can sit many commits apart, so stop this app's processes and start the checkout you mean.
- **A generation the app cannot serve.** `status` puts `ingest` in `requires`, and `status --verbose` reports `generation_upgrade_required` with the reason. The old generation stays searchable with a warning.
- **A project on slow storage.** Point `--runtime-root` at a fast device. The measured benefit is narrow; see `STORAGE.md`.
- **A search answers nothing.** Read `withheld_candidates`: it names the gate that dropped each candidate, which is usually the dense floor or the minimum passage length.
- **The UltraRAG runtime is broken.** `doctor` names the file; `doctor --repair-runtime` moves the old tree aside and installs the pinned one.

## UltraRAG credit and licensing

This app is built on the [UltraRAG](https://github.com/OpenBMB/UltraRAG) project through `vanilla-ultra-rag-mcp-server`, and uses the shared browser workspace package `ui-ultra-rag-mcp`. It depends on UltraRAG's MCP architecture, corpus chunker, and BM25 retriever, and adds PDF/EPUB extraction, project-scoped immutable generations, reviewed metadata, hybrid retrieval, and a browser workspace on top.

UltraRAG's upstream README identifies it as a joint project of THUNLP at Tsinghua University, NEUIR at Northeastern University, OpenBMB, and AI9stars, together with the wider UltraRAG contributor community. `NOTICE` records the upstream project, the pinned revisions, the models, and their licences.

This repository is an independent project. It is not an official UltraRAG release and is not affiliated with or endorsed by OpenBMB, THUNLP, NEUIR, AI9stars, or the UltraRAG contributors. The UltraRAG name is used only to identify the upstream software this app depends on.

The code in this repository is licensed under the Apache License, Version 2.0; see `LICENSE`. PDF and EPUB extraction depend on components under the AGPL, and `NOTICE` states what a recipient must satisfy. This repository's licence does not change those obligations.
