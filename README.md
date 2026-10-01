# research-rag

- `research-rag` is a research knowledge base over a project's own PDF and EPUB corpus.
- `research-rag` runs on your machine on one CPU, and no service is needed to sign up for.

## Purpose

- Point `research-rag` at a directory of originals, and it builds a searchable generation from them.
- Read every passage beside its source, its locator, and the metadata you reviewed.
- `research-rag` returns cleaned semantic evidence and never writes an answer.
- You conclude what the evidence supports.
- The untouched original is the only authority for a quotation.

## What it does

- **A browser workspace, an agent surface, and a command line** are three front ends to one process per project.
  - The only question a reader has to ask is whether the app is up.
- **Every search is hybrid and reranked.**
  - `MEASUREMENTS.md` carries the numbers, the weights, the gates, and the cost.
- **Review decisions survive a rebuild.**
  - A source's reviewed bibliography, categories, projects, keywords, language, title, and authors are a read-time overlay on the indexed corpus.
  - A rebuild never overwrites that overlay.
- **Exclusions are reversible.**
  - An excluded source leaves every retrieval surface at once.
  - The file holding that decision stays on disk.
- **Generations are immutable.**
  - A rebuild writes a new generation and switches to it only when every index is complete, so a failed build leaves the previous one searchable.
  - `generations` lists the generations and reclaims their space.
  - `STORAGE.md` carries the format.
- **Attached clients are visible and droppable from either place.**
  - The workspace status view lists every attached agent and ends one on request.
  - `clients` and `disconnect` do the same from a terminal.
  - A client that cannot open a socket gets the same surface through `research-rag mcp`, which speaks stdio and proxies to the app.
- **One client entry travels.**
  - A stdio entry names a project, not a directory.
  - The same file therefore serves that project on a laptop, a desktop, and a phone client, while the directory stays a fact of each machine.
- **One install serves many projects.**
  - Each project is a directory with its own corpus, app process, and port.
  - `init` records each project, `projects` lists each one, and `--project <name>` names one in place of its path.

`FEATURES.md` carries the full inventory, including what this app leaves out and why.

## Install

- Install the package on a machine with no checkout:

```bash
uv tool install git+https://github.com/AhmedKishki/research-rag.git
```

- Run it from a checkout:

```bash
git clone https://github.com/AhmedKishki/research-rag.git
cd research-rag
uv sync
```

- The `uv tool install` command needs `git` on the `PATH`.
- The `git clone` command needs `uv`.
- The pinned UltraRAG runtime and two models download on first use, which needs a network.
- `--offline` works once those files are cached.

## Put it on the command line and in the menu

- A checkout's console script lives in `.venv/bin/`, which no shell searches.
- `install` links the running interpreter's own script into a directory the shell already searches.
- A `uv sync` and an installed copy of the command both keep working after that link:

```bash
research-rag install
```

- A second run reports that the command is already installed and changes nothing.
- `research-rag install --uninstall` removes that link.
- `research-rag install --uninstall` refuses any path it cannot prove it wrote, so a script someone else placed there is left alone.
- One desktop menu entry per project starts that project's workspace and opens a browser:

```bash
research-rag --project-root /path/to/project install --desktop
```

- The entry runs `<project>/open-research-rag-ui.sh --open`.
- That launcher claims the port, records the pid, and stops the whole process group on `--stop`.
- `research-rag install --desktop --uninstall` removes the entry.
- `doctor` reports a menu entry whose project has been deleted, because that entry opens nothing.
- An entry this app did not write is reported and left alone unless you pass `--force`.
- Both files are written under the account's own directories: `~/.local/bin`, `~/.local/share/applications`, and `~/.local/share/icons/hicolor/scalable/apps/`.
- Nothing is written outside the account's directories.
- No project's own state is touched.

## First use

```bash
research-rag --project-root /path/to/project init --name "My project"
cp ~/Downloads/*.pdf ~/Downloads/*.epub /path/to/project/sources/
research-rag --project-root /path/to/project ingest
research-rag --project-root /path/to/project ui
```

- `init` records the project identity and writes the workspace launcher under `.research-rag/bin/`.
- `init` links that launcher into the project root as `open-research-rag-ui.sh`.
- `init` never overwrites a file it did not create.
- `status.ui_launcher` reports what `init` found.
- `ingest` extracts every included PDF and EPUB, chunks the text, embeds it, and builds a BM25 index and a dense index.
- The current generation stays active throughout an `ingest`.
- A second `ingest` call resumes a build that ran out of its time budget.
- Sources must be regular `.pdf` and `.epub` files in `sources/`.
- Markdown, symlinks, and anything outside `sources/` are ignored.
- Nothing in the app edits an original.

## Read the answer, then open the original

- A search answers with its passages and the `generation_id` they came from.
- A search carries a field only when it has news.
- `stale` means the corpus moved on.
- `rerank_fallback` means the reranker did not run.
- `generation_upgrade_required` means this generation cannot serve the request.
- An absent field is the ordinary case.
- **`text`** is cleaned semantic text for comprehension and paraphrase.
- **`text`** is never a transcript.
- **`source_relative_path`** and **`locator`** say where to open the original.
- The locator is the position alone: a page, or a section for an EPUB.
- A page carries `page_label` only where the printed label differs from the physical page.
- **`citation`** is the reference to attach to a claim.
- **`direct_quote_safe`** is `false` on every passage the app builds.
- Open the original at the locator and quote from it when you need exact wording.
- **`direct_quote_safe`** is the machine-readable form of that rule.
- Every search is hybrid and reranked because that mode beats every other measured mode on the judged set.
- A query carrying no topic word abstains rather than matching on a function word.
- A thin answer is a reason to ask again, not a conclusion.
- The reranker reorders about twice the number of passages you ask for, so a larger `top_k` deepens the ranking as well as the answer.
- `research-rag search` carries `--method` and `--no-rerank` only to reproduce a row of `MEASUREMENTS.md`.
- No reader-facing surface offers the choice.

## Narrow what a search reads

- Six filter layers are available, and every one is read from the reviewed metadata at query time:

```bash
research-rag --project-root /path/to/project search "commodity fetishism" \\
  --category "Commodity fetishism" --category marxism \\
  --keyword "fetishism of technology" \\
  --author Harvey --title "Fetish of Technology" \\
  --top-k 15
```

- Categories, projects, languages, authors, and titles match any of their values.
- Keywords match all of their values.
- Authors and titles match as case-insensitive substrings, because a name is a phrase rather than a controlled tag.
- A filter drops passages after the corpus is ranked, so the answer reports how much of the corpus the ranking reached: `filters.window_chunk_count` of `filters.corpus_chunk_count`, with `filters.window_is_whole_corpus` saying whether the ranking saw all of it.
- An empty result with a filter applied therefore means nothing the window reached matches that filter, not that no source carries it, and a filter naming one source can find nothing when that source ranks below the window.
- A filtered-out answer is not a broken index.
- `sources` is the inventory of the corpus.
- `passage CHUNK_ID --context-chunks 2` reads around a hit.
- `metadata`, `exclude`, and `include` record a decision.
- A metadata edit rewrites only that source's entry, so a hand edit to another entry in the same file survives.
- `exclude --chunk CHUNK_ID` excludes one passage on its own.
- A passage exclusion is applied when the answer is assembled rather than when the index is built, so the decision is enforced on the next search without a rebuild.
- Every search re-compares the source directory with the generation and reports `stale` with what changed: sources added, sources modified, sources gone, and whether reviews or exclusions moved.
- A reviewed metadata change is not staleness, because it is already effective.
- The complete payload reports `metadata_overlay_active`.
- Corrupt extraction units are omitted whole rather than indexed as garbage, and the count is reported.
- Script mixing never withholds a passage, because a foreign-language quotation inside an English source is evidence and its locator says where it is.

## Settings

- `research-rag config` prints every effective value, the layer it came from, what each key does, and what a change to each one costs.
- The workspace writes those values into `.research-rag/config.toml` after naming which keys force a rebuild.
- The workspace preserves every key it did not change, so a hand edit in that file survives.
- A value the environment, the command line, or `--config PATH` supplied is shown read-only, because those layers outrank the project file.
- Edit `.research-rag/config.toml` by hand to change a value outside the browser.
- Pass `--set key=value` to override a value for a single command.
- Pass `--config PATH` to add a layer.
- `research-rag help settings` names the layers, the description, and the cost.
- A corpus in a language the embedding model does not cover is reported rather than silently embedded.
- `research-rag help filters` names the filter layers.
- The example above shows the filter layers in use.

## Move between machines

- A project is self-contained: copy `sources/` and `.research-rag/`.
- `runtime/` is disposable and rebuilds on the next ingestion.
- Two processes pointed at one project root are refused by the project lock rather than silently interleaved.
- `STORAGE.md` carries the full layout, including `--runtime-root` for a project on slow storage.

## Connect an agent

- The app serves MCP at `/mcp` on its own port, so a client that can open a socket needs only the URL `http://127.0.0.1:<port>/mcp`.
- A client that speaks only stdio uses `research-rag mcp --project-name NAME`, which makes sure that project's app is up and then proxies to it.
- The stdio entry names a project and never a directory, so the same entry works on every machine where that project was initialised.
- `mcp` refuses `--project-root` and `--project`, `RESEARCH_ULTRARAG_PROJECT_ROOT` included.
- The `mcp` command comes before its own options, which is the only order the parser accepts: an option in front is read as the command name, and the server closes the connection instead of answering.
- On a machine holding no project under that name, the connection is made and `status` reports that the project is not initialised, with the `init` command that creates it.
- Running that `init` command serves the project with no edit to the entry.
- `RESEARCH_ULTRARAG_CLIENT_NAME` names the agent, so the app's client list can tell two agents apart.
- `RESEARCH_ULTRARAG_PROJECT_NAME` names the project for a client that can pass an environment but no argument.
- The port is chosen at start and recorded, so a hard-coded URL goes stale when the port moves.
- Print the entry for the machine it runs on, and check one already in place without editing it:

```bash
research-rag --project-root /path/to/project doctor --mcp-entry
research-rag --project-root /path/to/project doctor --check-entry ~/my-client.json
```

- Two ready-to-copy templates sit in this repository: `mcp_settings.example.json` for a client using an `mcpServers` object, and `kilo-mcp.example.jsonc` for one using a Kilo-style `mcp` object.
- Replacing the executable path and the project name in either template completes the entry.
- An agent gets eight tools and one resource, and every answer is the lean projection: one question, no inventory, and no scores.
- `status` is a verdict naming the call that closes a gap.
- `find_source` looks up one work by filename, title, or author.
- The full payload is `status --verbose` and the workspace.

## The command line

- `research-rag help` prints every command grouped by the work.
- `research-rag help TOPIC` explains one subject.
- Neither `help` form needs a project, so both work before you have one.
- Naming a command prints that command's own options.

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

- Every command takes `--project <name-or-id>` in place of `--project-root <path>`, so one shell can work on several projects.
- `--project` and `--project-root` together are refused rather than resolved by precedence.
- `status` is the one a person reads most, so it prints the lean verdict and takes `--verbose` for the whole payload.
- Every other command prints the whole payload.

## Diagnose an installation

- `doctor` prints one line per dependency: ok, warn, blocked, or unchecked, each with the command that fixes it.
- `doctor` writes nothing, so it is always safe to run.

```bash
research-rag --project-root /path/to/project doctor
research-rag --project-root /path/to/project doctor --prefetch-models
research-rag --project-root /path/to/project doctor --repair-runtime
```

- `doctor --prefetch-models` and `doctor --repair-runtime` are the only operations that reach the network.
- Neither of those two operations implies the other.
- A repair that discards evidence moves the evidence aside rather than deleting it.

## Update

- `research-rag update` checks for a newer version and writes nothing:

```bash
research-rag update
```

- `update` reports which kind of install it found, and compares that install with the repository's published releases rather than with a branch head.
- In a checkout, `update` names the version the checkout declares, the latest published release, whether the checkout is at it, behind it, or ahead of it with unreleased work, and what applying would do.
- In an installed distribution, `update` names the installed version, what is available, and the tool that owns the install.
- `update` finds that tool by asking `uv` and `pipx` rather than by guessing.
- A remote that cannot be reached, or a repository that has published no release, is reported as an answer, and nothing changed.
- `--apply` performs the update.
- `--apply` refuses while a build holds any project lock, and names the project, the phase that build is in, and the command that reports it.
- `--apply` refuses while the checkout has uncommitted work, and names the files in the way.
- Otherwise `--apply` stops every app this installation serves through that project's own launcher.
- `--apply` then reports the new revision, whether any project's portable state under `.research-rag` changed, and the exact command that starts each stopped app again.
- `--apply` does not start those apps for you.
- `--apply` prints the command that returns a detached checkout to its branch.
- `research-rag --version` prints this app's version, the version installed in the environment now, the shared workspace's version, and whether a restart is required.
- `update` reports the same four numbers from the same functions.
- `scripts/update.sh` is a wrapper around `research-rag update`: `--check` reports only, `--offline` does not touch the network, and a project path is accepted.

## Use the workspace

- A bare `research-rag` opens the workspace in a browser and serves it from the terminal you typed it in:

```bash
research-rag
```

```
AI and fetishism — workspace attached to /dev/pts/5
  http://127.0.0.1:5051
  pid 4080404 · log /path/to/project/.research-rag/runtime/logs/research-rag-ui.log
  Ctrl-C stops the app and the gateway it started.
```

- Ctrl-C in that terminal stops the app and the gateway it opened.
- Closing the window ends the app the same way, because a closed terminal sends the signal Ctrl-C would.
- Nothing survives that terminal.
- An installation holding more than one project is asked which one in the terminal.
- A terminal that cannot answer, a script or a pipe, is given the list of projects and the command to run instead of a prompt nobody will read.
- An app already up for that project is reported and left alone, because two apps on one project would each hold the lock and open a gateway.
- The app records the same state whichever way it was started, so another terminal can find and stop it.
- `research-rag projects` says whether each project's app is up and which terminal it is attached to.
- An attached app stops with its terminal, and a detached one does not.

```bash
research-rag --project-root /path/to/project ui            # start and print the URL
research-rag --project-root /path/to/project ui --open     # and open a browser
research-rag --project-root /path/to/project ui --stop     # stop it and what it started
./open-research-rag-ui.sh                                 # the generated launcher
```

- The launcher claims the first free loopback port at or above the one it was generated with and records the port it chose, so two projects never serve from the same port.
- `research-rag serve` is the same workspace in the foreground on a fixed port.
- The launcher runs `research-rag serve`.
- The workspace binds loopback only and has no authentication, which is correct for an address no other machine can reach.
- The workspace has no retrieval mode, no reranking switch, and no chunk tuning, because every search is hybrid and reranked.
- The workspace offers metadata, source selection, category partitions, project metadata, and this project's settings, each with the description the registry declares for it.
- The workspace has a tab per job: **Search**, **Sources**, **Config**, and **MCP**.
- The generations, partitions, languages, and SQL console panels sit under the search view.
- One installation serves several projects.
- The header names the project the page is serving and offers the others.
- A project whose app is up opens in a new tab.
- A project whose app is down shows the command that starts it rather than a link that would fail.
- The **MCP** tab carries the address this app serves agents on and the client entry `research-rag doctor --mcp-entry` prints, copyable as it is.

## Limitations

- **The app is a process, not a daemon.**
  - The app does not survive an update: stop it and start it again.
  - An agent attached to a stopped app has no session, and `clients` says so rather than listing a stale one.
- **Updates compare against a published release.**
  - `update` reads the remote's release tags, so the number it compares is the version a release carries rather than the branch head's latest commit.
  - A checkout ahead of the latest release reports unreleased work and changes nothing.
- **The first build is slow.**
  - The first build downloads the pinned runtime and two models, then extracts and embeds every source.
  - `status.ingestion_progress` reports progress, and a cancelled build resumes where it stopped.
- **`--offline` fails if anything is uncached.**
  - Run `doctor --prefetch-models` first.
- **A build reports "one build at a time".**
  - Another process holds the project lock, and `status` names the resident build's phase.
  - Reads are refused rather than queued while that build runs.
- **Two checkouts of this package.**
  - `doctor` reports `code_currency` as a warning.
  - Two copies of one version can sit many commits apart, so stop this app's processes and start the checkout you mean.
- **A generation the app cannot serve.**
  - `status` puts `ingest` in `requires`, and `status --verbose` gives the reason.
  - The old generation stays searchable with a warning.
- **A search answers nothing.**
  - `withheld_candidates` names the gate that dropped each candidate, which is usually the dense floor or the minimum passage length.
- **The UltraRAG runtime is broken.**
  - `doctor` names the file.
  - `doctor --repair-runtime` moves the old tree aside and installs the pinned one.
- **A workspace settings save rewrites the whole file.**
  - Comments in `.research-rag/config.toml` are not preserved, and keys are.
  - A save merges into what is there and changes only the keys you saved.

## UltraRAG credit and licensing

- This app is built on the [UltraRAG](https://github.com/OpenBMB/UltraRAG) project through `vanilla-ultra-rag-mcp-server`, and it uses the shared browser workspace package `ui-ultra-rag-mcp`.
- This app depends on UltraRAG's MCP architecture, corpus chunker, and BM25 retriever.
- This app adds PDF/EPUB extraction, project-scoped immutable generations, reviewed metadata, hybrid retrieval, and a browser workspace.
- UltraRAG's upstream README identifies it as a joint project of THUNLP at Tsinghua University, NEUIR at Northeastern University, OpenBMB, and AI9stars, together with the wider UltraRAG contributor community.
- `NOTICE` records the upstream project, the pinned revisions, the models, and their licences.
- This repository is an independent project.
- This repository is not an official UltraRAG release.
- This repository is not affiliated with or endorsed by OpenBMB, THUNLP, NEUIR, AI9stars, or the UltraRAG contributors.
- The UltraRAG name identifies the upstream software this app depends on and nothing else.
- The code in this repository is licensed under the Apache License, Version 2.0, and `LICENSE` carries that text.
- PDF and EPUB extraction depend on components under the AGPL, and `NOTICE` states what a recipient must satisfy.
- This repository's licence does not change those obligations.