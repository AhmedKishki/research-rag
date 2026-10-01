# research-rag

A research knowledge base over a project's own PDF and EPUB corpus. It runs on your machine, on one CPU, with no service to sign up for.

## Purpose

You point `research-rag` at a directory of originals, build a searchable generation from them, and read every passage beside its source, locator, and the metadata you reviewed. You conclude anything: the app returns cleaned semantic evidence and never writes an answer, and the untouched original is the only authority for a quotation.

## What it does

- **A browser workspace, an agent surface, and a command line**, all three front ends to one process per project. The only question a reader has to ask is whether the app is up.
- **Hybrid retrieval with reranking on every search.** `MEASUREMENTS.md` has the numbers, the weights, the gates, and the cost.
- **Review decisions that survive a rebuild.** A source's reviewed bibliography, its categories, projects, keywords, language, title, and authors are a read-time overlay on the indexed corpus, not something a rebuild overwrites.
- **Reversible exclusions.** A source leaves every retrieval surface at once, and the file stays.
- **Immutable generations.** A rebuild writes a new generation and switches to it only when every index is complete, so a failed build leaves the previous one searchable. `generations` lists them and reclaims their space; `STORAGE.md` has the format.
- **Clients you can see and drop, from either place.** The workspace status view lists attached agents and can end one; `clients` and `disconnect` do the same from a terminal. A client that cannot open a socket gets the same surface through `research-rag mcp`, which speaks stdio and proxies to the app.
- **One client entry that travels.** A stdio entry names a project, not a directory, so the same file serves the project on a laptop, a desktop, and a phone client; the directory is a fact of each machine.
- **Many projects, one install.** Each project is a directory with its own corpus, app process, and port. `init` records them, `projects` lists them, and `--project <name>` names one instead of repeating its path.

`FEATURES.md` has the full inventory, including what this app leaves out and why.

## Install

To install it as a package on a machine with no checkout:

```bash
uv tool install git+https://github.com/AhmedKishki/research-rag.git
```

To run it from a checkout:

```bash
git clone https://github.com/AhmedKishki/research-rag.git
cd research-rag
uv sync
```

The first command needs `git` on the `PATH`; the second needs `uv`. The pinned UltraRAG runtime and two models download on first use, which needs a network. After that `--offline` works.

## Put it on the command line and in the menu

The console script exists inside a checkout at `.venv/bin/`, which no shell searches. `install` links the running interpreter's own script into the directory the shell already searches, so a `uv sync` and an installed copy both keep working:

```bash
research-rag install
```

A second run says the command is already installed and changes nothing. `research-rag install --uninstall` removes that link and refuses any path it cannot prove it wrote, so a script someone else placed there is left alone.

One desktop menu entry per project starts that project's workspace and opens a browser:

```bash
research-rag --project-root /path/to/project install --desktop
```

The entry runs `<project>/open-research-rag-ui.sh --open`, which is the launcher that claims the port, records the pid, and stops the whole process group on `--stop`. `research-rag install --desktop --uninstall` removes the entry, and `doctor` reports an entry whose project has been deleted, because that entry opens nothing at all. An entry this app did not write is reported and left alone unless you pass `--force`.

Both files are written under the account's own directories: `~/.local/bin`, `~/.local/share/applications`, and `~/.local/share/icons/hicolor/scalable/apps/`. Nothing is written outside the account's directories, and no project's own state is touched.

## First use

```bash
research-rag --project-root /path/to/project init --name "My project"
cp ~/Downloads/*.pdf ~/Downloads/*.epub /path/to/project/sources/
research-rag --project-root /path/to/project ingest
research-rag --project-root /path/to/project ui
```

`init` records the project identity, writes the workspace launcher under `.research-rag/bin/`, and links it into the project root as `open-research-rag-ui.sh`. It never overwrites a file it did not create, and `status.ui_launcher` reports what it found.

`ingest` extracts every included PDF and EPUB, chunks the text, embeds it, and builds a BM25 and a dense index. The current generation stays active throughout. Call it again to resume a build that ran out of its time budget.

Sources must be regular `.pdf` and `.epub` files in `sources/`. Markdown, symlinks, and anything outside that directory are ignored, and nothing in the app edits an original.

## Read the answer, then open the original

A search answers with its passages, the `generation_id` they came from, and a field only when it has news: `stale` when the corpus moved on, `rerank_fallback` when the reranker did not run, `generation_upgrade_required` when this generation cannot serve the request. An absent field is the ordinary case.

- **`text`** is cleaned semantic text for comprehension and paraphrase. It is never a transcript.
- **`source_relative_path`** and **`locator`** say where to open the original. The locator is the position alone: a page, carrying `page_label` only where the printed label differs from the physical page, or a section for an EPUB.
- **`citation`** is the reference to attach to a claim.
- **`direct_quote_safe`** is `false` on every passage the app builds.

For exact wording, open the original at that locator and quote from it. The flag is the machine-readable form of the same rule.

Retrieval is not a choice you make, and that is deliberate. Every search is hybrid and reranked, because on the judged set that beats every other measured mode, and a query carrying no topic word abstains rather than matching on a function word. A thin answer is a reason to ask again, not a conclusion: the reranker reorders about twice the number of passages you ask for, so a larger `top_k` deepens the ranking as well as the answer. `research-rag search` carries `--method` and `--no-rerank` only to reproduce a row of `MEASUREMENTS.md`; no answer carries the choice.

## Narrow what a search reads

Six filter layers, all read from the reviewed metadata at query time:

```bash
research-rag --project-root /path/to/project search "commodity fetishism" \\
  --category "Commodity fetishism" --category marxism \\
  --keyword "fetishism of technology" \\
  --author Harvey --title "Fetish of Technology" \\
  --top-k 15
```

Categories, projects, languages, authors, and titles match any of their values; keywords match all of theirs. Authors and titles match as case-insensitive substrings, because a name is a phrase rather than a controlled tag. An empty result with a filter applied means no source matches it, and the answer names the filters it applied. A filtered-out answer is not a broken index.

Source selection narrows to named files: `sources` is the inventory, `passage CHUNK_ID --context-chunks 2` reads around a result, and `metadata`, `exclude`, and `include` record a decision. A metadata edit rewrites only that source's entry, so a hand edit to another entry in the same file survives. A passage can be excluded on its own with `exclude --chunk CHUNK_ID`, and the decision is enforced on the next search without a rebuild, because it is applied when the answer is assembled rather than when the index is built.

Every search re-compares the source directory with the generation and reports `stale`, naming what changed: sources added, modified, or gone, and whether reviews or exclusions moved. A reviewed metadata change is not staleness; it is already effective, and the complete payload reports `metadata_overlay_active`.

Corrupt extraction units are omitted whole rather than indexed as garbage, and the count is reported. Script mixing is never a reason to withhold: a foreign-language quotation inside an English source is evidence, and its locator says where it is.

## Settings

`research-rag config` prints every effective value, the layer it came from, what each key does, and what a change to each one costs. The workspace in the browser writes those values into `.research-rag/config.toml` after naming which ones force a rebuild, and it preserves every key it did not change, so a hand edit in that file survives. A value the environment, the command line, or `--config PATH` supplied is shown read-only, because those layers outrank the project file. To change a value outside the browser, edit that file, override one for a single command with `--set key=value`, or add a layer with `--config PATH`. `research-rag help settings` names the layers, the description, and the cost.

A corpus in a language the embedding model does not cover is reported rather than silently embedded. `research-rag help filters` names the layers; the section above shows them in use.

## Move between machines

A project is self-contained. Copy `sources/` and `.research-rag/`; `runtime/` is disposable and rebuilds. Two processes pointed at one project root are refused by the project lock rather than silently interleaved. See `STORAGE.md` for the full layout, including `--runtime-root` for a project on slow storage.

## Connect an agent

The app serves MCP at `/mcp` on its own port, so a client that can open a socket needs only the URL: `http://127.0.0.1:<port>/mcp`. A client that speaks only stdio uses `research-rag mcp --project-name NAME`, which makes sure that project's app is up and then proxies to it.

The stdio entry names a project and never a directory, so the same entry works on every machine where that project was initialised, and `--project-root` and `--project` are refused there, `RESEARCH_ULTRARAG_PROJECT_ROOT` included. On a machine that holds no project under that name, the connection is made and `status` reports that the project is not initialised, with the `init` command that creates it; run that command and the entry serves the project with no edit to it. Set `RESEARCH_ULTRARAG_CLIENT_NAME` to something that names the agent, so the app's client list can tell them apart, or `RESEARCH_ULTRARAG_PROJECT_NAME` when the client can pass an environment but no argument.

The port is chosen at start and recorded, so a hard-coded URL goes stale when it moves. Print the entry for the machine it runs on instead, and check one already in place without editing it:

```bash
research-rag --project-root /path/to/project doctor --mcp-entry
research-rag --project-root /path/to/project doctor --check-entry ~/my-client.json
```

Two ready-to-copy templates are in this repository: `mcp_settings.example.json` for a client using an `mcpServers` object, and `kilo-mcp.example.jsonc` for one using a Kilo-style `mcp` object. Replace the executable path and the project name in either and the entry is complete.

An agent gets eight tools and one resource, and every answer is the lean projection: a question at a time, no inventory, no scores. `status` is a verdict naming the call that closes a gap; `find_source` looks up one work by filename, title, or author. The full payload is `status --verbose` and the workspace.

## The command line

`research-rag help` prints every command grouped by the work, and `research-rag help TOPIC` explains a subject. Neither needs a project, so both work before you have one. Naming a command prints that command's own options.

| Command | What it answers |
|---|---|
| `projects` | every project this installation registered, and whether an app is serving it |
| `init` | the project identity, the app launcher, and the project's place in the register |
| `status` | the readiness answer, or the complete payload with `--verbose` |
| `ingest` | the current generation, a checkpointed build's progress, or a new complete generation |
| `search` | hybrid reranked evidence, with filters applied from reviewed metadata |
| `sources` | the corpus inventory, including files not indexed yet |
| `passage` | one passage and its neighbours |
| `include` / `exclude` | reversible retrieval decisions for a whole file or one passage (`--chunk CHUNK_ID`), enforced immediately |
| `metadata` | one source's reviewed bibliographic override |
| `config` | every effective setting, what the key does, and the layer it came from |
| `doctor` | one line per dependency, with the command that fixes it |
| `start` / `ui` | the app, started; `ui` also opens a browser |
| `clients` / `disconnect` | the agents attached to the app, and ending one |
| `mcp` | the agent surface on stdio for `--project-name`, proxied to that project's app |
| `serve` | the app in the foreground, which is what the launcher runs |
| `stop` | the app, and optionally any process of this app still building |
| `generations` | every generation on disk with its size, and the one search reads; `--use ID` searches a retained one instead |
| `remove-generation` | a generation search does not read, deleted after its id is repeated |
| `install` | the command on the account's `PATH`, and with `--desktop` one project in the desktop menu |
| `update` | what a newer version would change, and with `--apply` the update itself |
| `help` | every command grouped by the work, or one page of it |

Every command takes `--project <name-or-id>` in place of `--project-root <path>`, so one shell can work on several projects. The two together are refused rather than resolved by precedence. `status` is the one a person reads most, so it prints the lean verdict and takes `--verbose` for the whole payload; every other command prints the whole payload.

## Diagnose an installation

`doctor` prints one line per dependency: ok, warn, blocked, or unchecked, each with the command that fixes it. It writes nothing, so it is always safe to run.

```bash
research-rag --project-root /path/to/project doctor
research-rag --project-root /path/to/project doctor --prefetch-models
research-rag --project-root /path/to/project doctor --repair-runtime
```

The last two are the only operations that reach the network, and neither implies the other. A repair that discards evidence moves it aside rather than deleting it.

## Update

`research-rag update` checks and writes nothing:

```bash
research-rag update
```

It reports which kind of install this is, and compares it with the repository's published releases rather than with a branch head. In a checkout it names the version this checkout declares, the latest published release, whether the checkout is at it, behind it, or ahead of it with unreleased work, and what applying would do. In an installed distribution it names the installed version and what is available, and the tool that owns the install — `uv` or `pipx` — is found by asking each one rather than by guessing. A remote that cannot be reached, or a repository that has published no release, is reported as an answer: nothing changed.

`--apply` performs it. It refuses while any project lock is held by a build, naming the project, the phase it is in, and the command that reports it, and it refuses while the checkout has uncommitted work, naming the files in the way. Otherwise it stops every app this installation serves through that project's own launcher, then reports the new revision, whether any project's portable state under `.research-rag` changed, and the exact command that starts each stopped app again. It does not start them for you, and it prints the command that returns a detached checkout to your branch.

`research-rag --version` prints this app's version, the version installed in the environment now, the shared workspace's version, and whether a restart is required. `update` reports the same four numbers from the same functions.

`scripts/update.sh` remains a wrapper around this command: `--check` reports only, `--offline` does not touch the network, and a project path is accepted.

## Use the workspace

```bash
research-rag --project-root /path/to/project ui            # start and print the URL
research-rag --project-root /path/to/project ui --open     # and open a browser
research-rag --project-root /path/to/project ui --stop     # stop it and what it started
./open-research-rag-ui.sh                                 # the generated launcher
```

The launcher claims the first free loopback port at or above the one it was generated with and records the port it chose, so two projects never serve from the same port. `research-rag serve` is the same workspace in the foreground on a fixed port, and is what the launcher runs.

The workspace binds loopback only and has no authentication, which is correct for an address no other machine can reach. It has no retrieval mode, no reranking switch, and no chunk tuning, because every search is hybrid and reranked. Metadata, source selection, category partitions, project metadata, and this project's settings, each with the description the registry declares for it, are there. The workspace has a tab per job: **Search**, **Sources**, **Config**, and **MCP**, and the generations, partitions, languages, and SQL console panels sit under the search view.

One installation serves several projects. The header names the project the page is serving and offers the others: a project whose app is up opens in a new tab, and one whose app is down shows the command that starts it rather than a link that would fail. The **MCP** tab carries the address this app serves agents on and the client entry `research-rag doctor --mcp-entry` prints, copyable as it is.

## Limitations

- **The app is a process, not a daemon.** It does not survive an update; stop it and start it again. An agent attached to a stopped app has no session, and `clients` says so rather than listing a stale one.
- **Updates compare against a published release.** `update` reads the remote's release tags, so the number it compares is the version a release carries rather than the branch head's latest commit. A checkout ahead of the latest release reports unreleased work and changes nothing.
- **The first build is slow.** It downloads the pinned runtime and two models, then extracts and embeds every source. `status.ingestion_progress` reports progress, and a cancelled build resumes where it stopped.
- **`--offline` fails if anything is uncached.** Run `doctor --prefetch-models` first.
- **A build reports "one build at a time".** Another process holds the project lock, and `status` names the resident build's phase. Reads are refused rather than queued while it runs.
- **Two checkouts of this package.** `doctor` reports `code_currency` as a warning. Two copies of one version can sit many commits apart, so stop this app's processes and start the checkout you mean.
- **A generation the app cannot serve.** `status` puts `ingest` in `requires`, and `status --verbose` gives the reason. The old generation stays searchable with a warning.
- **A search answers nothing.** Read `withheld_candidates`: it names the gate that dropped each candidate, which is usually the dense floor or the minimum passage length.
- **The UltraRAG runtime is broken.** `doctor` names the file; `doctor --repair-runtime` moves the old tree aside and installs the pinned one.
- **A workspace settings save rewrites the whole file.** Comments in `.research-rag/config.toml` are not preserved; keys are. A save merges into what is there and changes only the keys you saved.

## UltraRAG credit and licensing

This app is built on the [UltraRAG](https://github.com/OpenBMB/UltraRAG) project through `vanilla-ultra-rag-mcp-server`, and uses the shared browser workspace package `ui-ultra-rag-mcp`. It depends on UltraRAG's MCP architecture, corpus chunker, and BM25 retriever, and adds PDF/EPUB extraction, project-scoped immutable generations, reviewed metadata, hybrid retrieval, and a browser workspace.

UltraRAG's upstream README identifies it as a joint project of THUNLP at Tsinghua University, NEUIR at Northeastern University, OpenBMB, and AI9stars, together with the wider UltraRAG contributor community. `NOTICE` records the upstream project, the pinned revisions, the models, and their licences.

This repository is an independent project. It is not an official UltraRAG release and is not affiliated with or endorsed by OpenBMB, THUNLP, NEUIR, AI9stars, or the UltraRAG contributors. The UltraRAG name is used only to identify the upstream software this app depends on.

The code in this repository is licensed under the Apache License, Version 2.0; see `LICENSE`. PDF and EPUB extraction depend on components under the AGPL, and `NOTICE` states what a recipient must satisfy. This repository's licence does not change those obligations.
