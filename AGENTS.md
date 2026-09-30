# AGENTS.md

This is the engineering guide for AI coding agents working in `research-rag`.


## Objective
Serve a project-scoped research knowledge base built from original PDF and EPUB sources, over one loopback port, from one process. A person opens the workspace, searches their corpus, reads each passage beside its source, its locator, and its reviewed bibliography, and records what they decide about each source. An agent retrieves the same passages through the same service. The app returns cleaned semantic evidence across a collection and preserves document identity and original-file locators. It never writes an answer: the reader is the one who concludes anything, and the untouched PDF or EPUB is the quote authority.

One process owns the project, its lock, and its UltraRAG gateway. The browser workspace, the agent's tools and resources, and the command line are three front ends to that one instance, not three processes that each build their own view. An agent connects to the running app at `/mcp`, or through the stdio bridge for a client that speaks only stdio. The app can list and disconnect the agents attached to it. The only question a reader has to ask is whether the app is up.

The package builds on the separately versioned `vanilla-ultra-rag-mcp-server` for UltraRAG's corpus chunker and BM25 retriever, and on the separately versioned `ui-ultra-rag-mcp` for the browser workspace. Never add research behaviour to either of those to support this project.

## Documentation responsibilities
- `README.md` is a standalone user manual: capability summary, operation, installation, concrete usage, expected results, storage, and user-visible limitations. It must not compare or link to sibling projects in this collection. It must document every installed command, including the ones an agent or a browser uses, because all three are how a person reaches the same project.
- `STORAGE.md` is the state format: every file this app writes, what it means, which is portable, and which is rebuilt. It is the document a person reads before editing a project by hand, and it is where the field names below are defined.
- `AGENTS.md` is this engineering contract. It is the single place for current engineering rules, and it may document internal dependency boundaries, but it must not become a second user manual.
- `FEATURES.md` is the capability inventory: what comes from UltraRAG, what this app adds on top, what is deliberately excluded, what is only planned, and how it differs from a general-purpose RAG server. Keep the upstream/added/planned labels accurate; never present planned work as shipped. Whenever UltraRAG offers a capability this app does not use, section 1.1 must state the reason, what reuse would have added, and the criteria under which the decision would be revisited; an unused upstream feature must never appear unexplained.
- `MEASUREMENTS.md` holds the current numbers and the current limits: ingestion and query cost, retrieval quality against the judged set, and what those numbers do not establish. Every measurement quoted anywhere else in the repository is reproduced there.
- `ROADMAP.md` contains only deferred product ideas that are not in scope yet.
- `TODO.md` contains only open work: what is unimplemented, grouped by the problem each item solves, with the commands that verify the repository's current state.
- `NOTICE` contains attribution, provenance, and legal notices.

### Markdown describes the present, never the past

Every markdown file states what the software does now and what remains open. Git is the archive, and its history is the only record of how the current state was reached.

- Do not add review logs, finding lists, change histories, before/after narratives, "Step N" sequences, decision logs, or "previously"/"used to"/"was" framing. If a fact changed, state the current fact and let the diff carry the rest.
- Do not keep a finished item as a record of itself. Once work is done it leaves `TODO.md`; once a measurement is superseded, the superseded number leaves `MEASUREMENTS.md`.
- Keep numbers current. When the corpus, the extraction policy, or a retrieval default changes, re-measure with the harnesses named in `MEASUREMENTS.md` and update every quoted figure rather than leaving one stale value that contradicts the rest.
- Keep each fact in one place. Current rules live in this file, current behaviour in `README.md`, the state format in `STORAGE.md`, capabilities in `FEATURES.md`, numbers and limits in `MEASUREMENTS.md`, open work in `TODO.md`, and deferred ideas in `ROADMAP.md`; other files point at them instead of repeating them.
- Cross-references must resolve after any edit: code comments, scripts, and documents name these files by path or by harness name, never by a section number that can drift.

### Only direct, concise language

Write every file — documents, commit messages, code comments, docstrings — in direct, concise language. State the fact and stop.

- One sentence per fact. No preamble, no hedging, no restating what the reader already read.
- Cut filler: *very*, *really*, *actually*, *simply*, *just*, *it is worth noting*, *importantly*, *of course*.
- Do not sell. No superlatives or dramatic framing; a number, a mechanism, or a limit is the argument.
- Prefer a concrete noun and an active verb: "the build fails" over "a failure condition may be encountered".
- Length is not thoroughness. Split long sentences, turn paragraphs into bullets, and never open an item by restating its own title.

## Presenting decisions to the user
Any question that needs a user choice must be presented as a numbered list of concrete options, never as an open question. For every decision:

- give each option a short identifier (`A`, `B`, `C`, …), a one-line statement of what it does, and the smallest change it requires;
- state pros and cons for each option separately, covering cost, risk, and the effect on the research contract;
- state whether each option is reversible and what undoing it would take;
- mark exactly one option as the recommendation and say why in one sentence;
- include an explicit "no change" or "decide later" option whenever work can proceed without an answer;
- keep options mutually exclusive, and complete enough that choosing one is sufficient to proceed;
- keep no decision log: a choice becomes a current rule in this file when it changes behaviour, or an entry in `TODO.md` or `ROADMAP.md` when it defers work;
- never implement a choice that changes generation artifacts, retrievable evidence, the portable-state contract, or on-disk layouts before the user has chosen it.

## Current compatibility baseline
- Package: `research-rag`
- Command: `research-rag`, with `init`, `status`, `ingest`, `search`, `sources`, `passage`, `include`, `exclude`, `metadata`, `config`, `doctor`, `start`, `ui`, `clients`, `disconnect`, `mcp`, `serve`, and `stop`
- Agent surface: seven tools and one resource at `<app>/mcp` on the app's own port, plus the stdio bridge for a client that cannot open a socket
- Version: `0.1.0`
- On-disk compatibility: byte-compatible with `research-ultra-rag-mcp`, which is frozen and still installed on the machines that carry it, so the two keep reading and writing the same projects. The names that are inherited rather than this app's own are `USER_CONFIG_DIRECTORY`, `SETTINGS_ENVIRONMENT_PREFIX`, the model-cache directory, and the `runtime.tool_detail` key; `tests/test_data_roots.py` states each one and why it is retained. This app is the only one of the two that changes.
- `bm25s` is pinned to a fork (`AhmedKishki/bm25s` @ `20f6c02`) carrying a one-line fix for its non-ASCII stopword serialization; `settings` fails fast on a stopword list that cannot round-trip through it, so revert the pin only once upstream fixes it.
- `pymupdf` raises `IndexError` from `Page.get_label()` when a document's page-label tree starts after the page being asked about, which fails an entire extraction and so an entire ingestion; reported upstream at https://github.com/pymupdf/PyMuPDF/issues/5140. `_pdf_locator` catches it and falls back to the physical page number, and two tests pin that a real label still wins. Narrow the guard when that issue closes.
- Licence: Apache-2.0 for this repository's own code (`LICENSE`); `NOTICE` records the upstream UltraRAG, model, retrieval-component, and AGPL-3.0 extraction-dependency terms, which stay separate from that grant.
- Python: `>=3.11,<3.13`
- FastMCP: `3.4.0`, used only to reach the pinned vanilla gateway
- Vanilla gateway commit: `fc339c259a672ca4dacb525840eba851d01c4b75`
- Shared UI commit: `98a97a9`
- Shared settings-core commit: `cbd47bb46efe85340de34f8f8f13fc6516e7fecb`
- Upstream UltraRAG: `0.3.0.2` at `3a709a2aea3fbe46acca59c422621c94b6e86857`

## Non-negotiable research contract
- Retain the prominent UltraRAG acknowledgement in `README.md`, the root `NOTICE`, upstream project links, license information, and the independent project disclaimer.
- Credit THUNLP, NEUIR, OpenBMB, AI9stars, and the upstream contributors using the wording supported by UltraRAG's own README. Do not imply endorsement.
- One app process serves exactly one configured project root, and every surface it serves shares that process, its project lock, and its single UltraRAG gateway.
- Every project's workspace must be reachable and identifiable on its own: the generated launcher claims the first free loopback port at or above the one it was generated with and records the port it chose, never a fixed port two projects can collide on, and each workspace names the project it serves rather than a generic label.
- The app exposes one agent surface: the seven operations and the one resource, mounted at `/mcp` on the same loopback port as the workspace, plus the stdio bridge that proxies to it. Both must keep the same capability the previous server had, and neither may re-declare a tool: a second copy of an operation is a second place for it to be wrong.
- The app is up or it is down, and that is the one thing to check. A command reaches the running app over the control API; a project with no app up is answered in process, because a read of local state must not refuse itself on the grounds that a daemon is not running.
- Never accept an arbitrary filesystem output path.
- Ingestion selects only regular PDF and EPUB files beneath the configured sources directory.
- Reject source symlinks and path traversal.
- Store every project-owned artifact beneath `<project>/.research-rag` by default: portable identity and review state at its root and all disposable derived state beneath `.research-rag/runtime`. An explicit `--runtime-root` may relocate derived state only, and only to an absolute path claimed by a marker naming this project's `project_id`; review state must stay in the project, and two projects must never share one runtime root. The single exception outside that directory is the machine-local `<project>/open-research-rag-ui.sh` symlink to the generated launcher beneath `.research-rag/bin/`; nothing else belongs outside `.research-rag`.
- Every path this app names inside a project carries `research-rag`, and the launcher's state files are named `research-rag-ui.pid`, `.port`, `.log`, and `.lock` for the same reason. The frozen `research-ultra-rag-mcp` generates its launcher beside this one in the same project, and two products sharing a pid file would let each stop the other's process.
- Share only immutable model binaries through the configured user cache. Never place documents, metadata, chunks, vectors, indexes, logs, or query state in global storage.
- Never edit or write the original source documents.
- Keep source exclusions explicit, reversible, project-local, and immediately enforced by every retrieval surface. Do not add automatic duplicate guessing.
- Do not switch `current.json` until a generation is completely indexed.
- Preserve deterministic source IDs, document IDs, chunk IDs, source paths, and locators. A project-scoped `source_id` is based on the normalized source-relative path, survives byte changes, and changes on rename/move. A `document_id` identifies a path/content version. Chunk IDs may change when content or chunking configuration changes; never imply that document or chunk IDs are permanent across incompatible generations.
- Keep BM25 and dense indexes in the same immutable generation, and never select the generation unless both indexes validate successfully.
- Never let a checkpoint, `current.json`, or any other commit point become durable before the artifacts it describes. Grouping directory fsyncs inside a unit is required and expected (`fsync_directories` before the checkpoint write); dropping the artifact write's own durability, the checkpoint's, or the ordering is not. A file written only to hand off to a peer process and rewritten before every use may skip fsync entirely.
- Keep dense vectors and any dense index project-local; do not introduce a required external database service.
- Record the dense backend in each generation manifest and dispatch retrieval from that record. Never rebuild an existing generation with a different backend, and never assume a fixed dense index directory name.
- Search results present `text` as cleaned semantic text and never as a transcript, and direct quotations must come from the original. State that once for a reader in `README.md` and once as the field's own description in `STORAGE.md`; never as a per-answer field, a workspace label, or a footer, and never per passage. The `direct_quote_safe` flag stays in every payload.
- Keep every tunable in the registry in `settings.py`, and every default value in the packaged `default.toml`. Code reads a value from the resolved settings; it never keeps a second copy of a default. A layer names only the keys it changes, an undeclared key is refused in every layer, and `research-rag config` reports each effective value with the layer that supplied it.
- The layer machinery is not this repository's. `Setting`, the merge, the coercion, the provenance, the three path helpers, and `describe_settings` live in the separately versioned `config-ultra-rag-mcp` library, pinned by commit, which other products in this collection use as well. This repository owns `SETTINGS` and its three derived maps, `EffectiveSettings` and the rules only its values can break, the packaged `default.toml`, and the three names it resolves its own layers by. It imports the rest; `tests/test_architecture.py` fails if a copy of the machinery reappears here. Skew between this app's pin and another product's is allowed and is not a bug: a floating or vendored dependency is what would break the collection's rule that each member is independently installable.
- Keep schema versions, policy versions, the retrieval-method set, and the boundary names in code. Those define what a generation is, or where the app may write, and a settings file must not be able to forge either. The test for a tunable is the identity rule: if its value decides what a generation contains, it enters the retrieval-policy fingerprint or the recorded chunk settings; if it cannot change an artifact, it is a runtime setting.
- One answer shape per operation: the complete service payload, and a projection of it for the bounded readers. The workspace always wants the whole thing, because the ranking scores, the candidate counts, and the withheld-candidate reasons are what explain an empty answer to the person reading it, and it must never project. Two readers are bounded and share `tool_views`, so a terminal and an agent cannot disagree: the agent's tool answer goes through `present_tool_response`, and `research-rag status` prints `lean_status` unless `--verbose` asks for the payload. `runtime.tool_detail` selects the agent's mode, and it is the one setting that is not a project, corpus, or machine tunable: a settings file the MCP server wrote names it, and the layer stack refuses an undeclared key in every layer.
- One install serves many projects, and the machine remembers them. Each project is a directory with its own `.research-rag`, its own app process, its own port, and no shared corpus; `registry.py` holds the only machine-wide record, an id, a name, and a root per project in the account settings directory, written by `init` and replaced atomically. It is a pointer file: no corpus, no state, nothing a deleted project can lose. `--project <name-or-id>` resolves it and is refused alongside `--project-root` rather than resolved by precedence, because a command that ran against the wrong project is worse than one that did not run. `projects` reports the register without starting anything. Never let a pointer file grow into a cache of a project's state, and never let one project's record decide what another project may read.
- An agent's tools answer one question at a time, and no tool lists the corpus. `status` is the verdict (`ready`, `stale`, `requires`) and the conditions to act on; `find_source` resolves one filename, title, or author and returns that source's handle and whether it is searchable. The workspace keeps the full inventory and the command line prints it through `sources`, because a person reading the inventory is the reader those answers serve. Never grow an agent answer toward the workspace's, and never let a tool answer scale with the size of the corpus.
- Every operation takes only the arguments its reader must decide, and no surface takes what another surface will not honour. The agent's tools offer no retrieval mode, no output view, and no chunk tuning, because a capability the measurements already answer (hybrid retrieval, reranking, the freshness check, chunking) is an engine setting the app fixes. The workspace's profile turns off what it does not serve and its adapter drops every argument an operation does not accept. The engine keeps the switches for the harness and tests. `search` is always hybrid, always reranked, and always staleness-checked.
- One recorded exception to that rule: `research-rag search` keeps `--method {bm25,dense,hybrid}` and `--no-rerank` so a row of `MEASUREMENTS.md` can be reproduced on demand. The workspace exposes neither and no answer carries the choice, because a reader has nothing to decide with them. Removing the flags is a deliberate simplification, not an oversight; deleting them without deciding that is not.
- Keep the corpus language and the embedding model as settings, and keep each model's facts in its pinned table rather than beside its use. `language.corpus` selects the BM25 stopwords and enters the ranking policy fingerprint; `dense.embedding_model` selects a model whose dimension, revision, token limit, covered languages, and required prefixes come from `embeddings.py`. A corpus language the model does not cover is reported, never silently embedded.
- Every search reranks, and the workspace shows no switch, because reranking is the largest measured quality gain. The model comes from the pinned table in `rerankers.py`, may be selected by an operator (`--reranker-model`) or per call by the engine (`search(rerank_model=...)`), and is named with its revision in the payload. An unavailable model degrades to the unranked candidate order with `rerank_fallback`, never a failed search. Adding a model to that table means pinning its revision, and never resolving a model name at run time.
- A search names a source only when the reader has to act on it — a removal, an exclusion, a review, or the source a passage came from. `sources` is the inventory, and no other answer becomes a corpus listing: a `status` answer carries no source, chunk, or document count because the rows that count them are one operation away.
- The shared workspace is pinned and generic, so its profile turns off every capability this app does not serve (`retrieval_modes=False`, `reranking=False`, `chunk_settings=False`) and `ResearchUIAdapter._arguments` drops every argument an operation does not accept. The pinned workspace sends its full optional set for every route, so the adapter is the only place a control the app does not serve can be refused. Never let a workspace control travel as an argument the app ignores. Metadata and its filters stay on because the app serves both: reviewed metadata is a first-class feature and `search` accepts the filter layers.
- Each surface names its own arguments, and no surface imports another. That is what keeps a workspace control, an agent's schema, and a command line from becoming three copies of one operation, and it is enforced rather than trusted: `tests/test_architecture.py` fails if a surface imports another or if a second file declares a tool or a resource.
- Name every condition a reader must act on, and put the remedy beside it. `health.py` owns one named check per condition, each in one of four states: `ok`, `warn` (the answer is worse), `blocked` (the app cannot work), and `unknown` (the check did not run, which never reads as healthy and is never disclosed as fine). Every check reads: no process, no network, no write, so the report can be built before the gateway is opened. A condition the report cannot state must read as unknown rather than be omitted.
- Report a condition at the moment it appears, not when a later stage trips over it. A `status` answer carries `blocked_by` and `degraded`, each a list of `{check, reason, remedy}`, both omitted when empty and both present together when a project is healthy; the per-check detail is a `doctor` reader, because a status answer discloses a condition rather than inventorying checks. A reader who sees `blocked_by` must be able to name the fix without reading this file, so a reason is a sentence and a remedy is a command that can be pasted.
- The `doctor` command reads that same report, so the two surfaces cannot disagree. It writes nothing at all, and treats `--prefetch-models` and `--repair-runtime` as the only operations that reach the network; it must not imply one by accepting the other. A repair that discards evidence moves it aside instead of deleting it, and a check that could not run is printed as unchecked.
- A failure message must name the reason and where to read the rest. A gateway that cannot start, and a call that never answered, answer with the tail of the logs the transport already writes plus the path of each one, so `Connection closed` is never the whole of what a reader is told. No process-tree inspection: the vanilla tool name says which component was busy, and that component writes its own log.
- The vanilla gateway installs its verified runtime read-only. This app may depend on a pinned release that predates that, and must degrade to naming the file only when the release can: a mismatch is `blocked` either way, and the reason says so when the release cannot name the path. `doctor --repair-runtime` is what automates the advice the runtime's own error message gives.
- Keep the workspace bound to loopback addresses. Do not add remote exposure or authentication assumptions without an explicit security design.
- `App` claims its port by binding it before uvicorn is created: a claim is never handed out twice, a failed claim is reported through `error` with its reason instead of being raised or silently empty, and a released port is claimable again because no verdict is cached. The generated launcher is what chooses a port and opens a browser; `serve` takes the port it was given and never chooses one, so the launcher and a manual run cannot both be picking ports for one project.
- One port, three surfaces. The agent endpoint and the workspace are composed into one ASGI application and dispatched by path, because mounting the agent app under its own path would prefix its own routes twice, and a catch-all mount does not fall through on a 404. Each mounted application owns state its own handlers read back, and Starlette does not run a mounted application's lifespan, so both are entered by the app's own lifespan: the agent app's for its session machinery, which is what makes the endpoint answer a session at all, and the workspace's to install the adapter it was handed.
- The workspace reports and ends the app's clients, through the pinned shared UI's opt-in `clients` capability and its two adapter methods, over the same registry the command line reads. A drop from the browser is a drop the command line can see, and it must stay that way: a second registry per surface would make the surfaces disagree about who is attached. A workspace served without a process behind it refuses with a 501 rather than reporting an empty list, because "no agents are attached" and "there is no app" are different facts.
- An attached agent is a client with a session. A connection that has not carried one yet is recorded as a sighting, never counted as attached, and reconciled onto the session when the id appears. A disconnect refuses that session before the request reaches the tools, and drops the client's sightings by the name the client gave itself, because the MCP SDK opens a notification stream before the session exists and a stream that cannot be matched would leave the client hanging instead of ending.
- Every child process this app starts is built with `child_process_environment()`, which drops the variables in `TOP_LEVEL_ONLY_ENV` and marks the child with `MANAGED_CHILD_ENV`. The only child is the pinned vanilla gateway. Never hand a child `dict(os.environ)`, and never point this app at a second copy of itself: two apps on one project would each hold the project lock and each open a gateway, which is the duplication this product exists to remove.
- Keep browser and control write endpoints same-origin, JSON-only, and constrained to the operations the app serves. The control API is the command line's channel to the app and is loopback-only for the same reason the workspace is.
- Do not let workspace code read or mutate generation artifacts directly. It must call `ResearchService`, except for safely serving an allowlisted original PDF or EPUB from the configured source root.
- Keep the shared UI dependency pinned by commit. Keep the adapter, source authorization, and the process host in this repository; do not copy the shared static workspace back into this package.
- The on-disk contract is frozen while `research-ultra-rag-mcp` is installed. That product is frozen and will not be changed, and it reads and writes the same `.research-rag` directory, so a schema-version or field change here that it cannot parse is the fault class a stale copy already caused once: a reader answering with state it could not have written. Change it only with a version it also understands, because a change this app makes alone leaves every machine running both products with a project neither can be trusted to read.
- Initialising a project creates a machine-local launcher and links it into the project root: the script lives at `.research-rag/bin/open-research-rag-ui.sh` and `<project>/open-research-rag-ui.sh` is a relative symlink to it. Create both only when absent, never overwrite a file or symlink this app did not create, never let a filesystem failure stop startup, and report the state in `status.ui_launcher`. The generated script must start the workspace with the project's own `--project-root`, `--runtime-root`, and `--port`, must resolve the app's console script beside the running interpreter rather than trusting `PATH`, must pass explicit flags rather than exporting settings, and must stop the whole process group on `--stop`. Probing for a free port and binding it are two steps, so the script chooses its port while holding an atomically created lock under the state root, and it records the pid and the port only once the port is served by the process it started — another project's workspace answering on that port is exactly what must not count, and a workspace that lost the bind is on its way out. A start that fails moves to the next port (an explicit `--port` is final), and `--stop` refuses to signal a pid whose command line does not name this project.

## Architecture
```text
MCP client ── HTTP /mcp ─┐
                       │
stdio client ── bridge ─┤
                       ├── research-rag (one process, one port, one lock)
Local browser ── HTTP ──┤      │
                       │      ├── ResearchService
Terminal ── control ───┘      ├── shared workspace (surfaces/ui.py)
                              ├── agent tools and resources (surfaces/mcp.py)
                              ├── control API (control.py)
                              ├── client registry (app.py)
                              │
                              ├── project/source policy
                                                          ├── PDF page extraction
                                                          ├── EPUB section extraction
                                                          ├── metadata and provenance
                                                          ├── immutable generations
                                                          ├── FastEmbed CPU embeddings
                                                          ├── project-local dense index
                                                          ├── reciprocal-rank fusion
                                                          ├── CPU cross-encoder reranking
                                                          └── stdio MCP -> vanilla-ultra-rag-mcp
                                                          ├── UltraRAG corpus chunker
                                                          └── UltraRAG BM25 retriever
```

One process serves the workspace, the agent, and the command line. The vanilla gateway is an implementation dependency below it, not a second surface; the shared workspace package is loopback HTTP infrastructure that owns no research state; and the stdio bridge is a transport, not a second copy of the tools.

## Intentional differences from Vanilla RAG
The official UltraRAG Vanilla RAG pipeline includes benchmark loading, dense retrieval, prompt rendering, model generation, answer extraction, and evaluation. This app reuses UltraRAG's chunking and retrieval mechanics but changes the boundary for research work:

- PDF/EPUB extraction, project isolation, immutable storage, metadata, and locators are implemented here.
- Retrieval combines UltraRAG CPU BM25 with a project-local FastEmbed dense index. This extension owns fusion, filters, scores, and provenance.
- `search` returns visible, structured semantic evidence instead of anonymous passage strings. It is explicitly not an exact-quotation surface.
- The reader is the generation stage. This app calls no UltraRAG generation internally and has no chat surface.
- Benchmark loading, boxed-answer extraction, and automatic evaluation are not part of the interactive research flow.

Do not blur this boundary in documentation. Adding answer generation would be a deliberate research feature requiring its own API, citation contract, tests, and user-visible model configuration.

## Repository map
- `app.py`: the running app. One project, one port, one service, the composed ASGI application, the MCP client registry, and the port claim.
- `control.py`: the control API the command line speaks, and the `Control` handle it speaks it with, plus `ensure_running` for the command that needs the app.
- `bridge.py`: the stdio front end, which makes sure the app is up and proxies stdio to its agent endpoint under a name it reports.
- `surfaces/cli.py`: the command line — the command centre for the app's lifecycle, its clients, its corpus, and the projects it serves.
- `surfaces/mcp.py`: the agent surface — the seven tools, the one resource, and the answer projection.
- `surfaces/ui.py`: the workspace profile, the adapter that maps its operations to the service, and safe original-source authorization.
- `tool_views.py`: the lean/full answer projection, used by the agent surface and by the command line's default `status`.
- `registry.py`: the account's record of the projects one installation serves — id, name, and root — beside the settings file, and nothing else.
- `instructions.py`: what an agent is told about the project, the order of work, and what it is owed.
- `config.py`: project boundary, stable identity, executable validation, and the names this app resolves its own layers by.
- `sources.py`: allowlist, source discovery, hashing, and metadata validation.
- `extraction.py`: layout-aware PDF/EPUB extraction and bibliographic identity.
- `artifact_lookup.py`: generation-local SQLite offsets for selective canonical chunk/unit reads and exact-text vector reuse without copied corpus text.
- `storage.py`: atomic JSON state and JSONL artifacts.
- `launcher.py`: the generated per-project launcher, its project-root link, and its state file names.
- `version.py`: the version this process started with, the installed version, the shared-UI version, the checkout it came from, and the `status.version` block and workspace header label they feed.
- `dense.py`: pinned FastEmbed models, both local dense backends (exact scan and embedded ANN) with document filtering, and the CPU cross-encoder that reranks every search.
- `rerankers.py`: the pinned reranker-model table and its revision resolver, so a model choice is a lookup rather than a download by name.
- `embeddings.py`: the pinned embedding-model table, including each model's dimension, token limit, covered languages, and required prefixes.
- `settings.py`: the tunable registry and the effective settings it validates, resolved over the shared layer stack.
- `default.toml`: the packaged default for every tunable, one commented entry per setting.
- `generation.py`: exact compatibility checks and validated reuse snapshots.
- `ultrarag.py`: the pinned vanilla gateway, the `LazyGateway` that opens it on first use, and the log a failed start or a timed-out call names.
- `health.py`: the named dependency checks, one per condition a reader must act on, built from a status payload and read by both `status` and `doctor`.
- `doctor.py`: the `doctor` command — the report and the two flags that reach the network.
- `service.py`: the project-scoped service itself — the four workflow mixins combined into `ResearchService`, the project lock, the project paths, and the module's public names.
- `ingestion.py`: resumable ingestion — staging, checkpoints, activation, and recovery.
- `search.py`: ranked retrieval — BM25, fusion, reranking, and evidence assembly.
- `status.py`: readiness, staleness, upgrade reasons, the generation inventory, and the health report that reaches the `status` answer.
- `review.py`: source inventory, reviewed metadata, and reversible exclusions.

- `ui-ultra-rag-mcp` dependency: loopback HTTP host, constrained JSON API, and packaged dependency-free browser workspace.
- `scripts/benchmark_write_pattern.py`: reproduces the write, chunking, and embedding measurements in `MEASUREMENTS.md`.
- `scripts/evaluate_retrieval.py`: runs the judged query set through `ResearchService.search` in process, and reports BM25, dense, hybrid, and reranked quality, because the app itself is hybrid-only; its JSON report is a generated, gitignored artifact.
- `scripts/update.sh`: pulls the checkout, syncs the environment, optionally restarts a named project's workspace, and reports the process restart it cannot perform itself.
- `evaluation/`: the reference judged query set and its protocol, including the known-item limits that keep it from claiming true recall.
- `tests/`: unit and real-gateway integration coverage.
- `ROADMAP.md`: explicitly deferred work.

## Storage model
Portable, project-owned state lives at:

```text
<project>/.research-rag/project.json
<project>/.research-rag/source-catalog.json
<project>/.research-rag/source-metadata.json
<project>/.research-rag/source-exclusions.json
```

`project.json` is authoritative for the stable project ID, name, and project-relative source directory. CLI entrypoints reuse its source setting when `--source-directory` is omitted; an explicit differing value must fail.

Disposable local state lives at:

```text
<project>/.research-rag/runtime/current.json
<project>/.research-rag/runtime/project.lock
```

`--runtime-root` (or `RESEARCH_ULTRARAG_RUNTIME_ROOT`) may relocate the whole runtime root, including `current.json` and `project.lock`, to an absolute path outside the project, which is how a project on slow storage keeps generations, staging, and index builds on a fast device. The root is claimed on first use by a `.research-ultra-rag-runtime.json` marker holding the owning `project_id`; a root with a different `project_id`, a non-empty root with no marker, a relative path, the project root, and a non-directory are all rejected with explicit messages. Review state in `.research-rag` never moves, and the legacy-location migration runs only for the default root.

`resolve_config` performs a guarded one-time move from the legacy `.ultrarag/research` location. Never create new research state there. Refuse to guess when both old and new locations contain runtime payloads. Migration must run only after older research MCP and UI processes have stopped.

Changed builds use a unique directory under `staging/`, then move a verified generation beneath `generations/` before switching `current.json`. Successful generations retain only the manifest, cleaned extraction units, final chunks, portable float32 vectors, a text-free SQLite offset lookup, BM25 index, and dense index selected for that generation. UltraRAG raw chunks are temporary staging data, and raw coordinate records are not generated. Bounded calls, cancellations, and timeouts retain an atomic `checkpoint.json` and only committed work; incompatible inputs supersede that checkpoint with a small diagnostic. Non-resumable failures remove heavy staging data and leave a small record under `failures/`. Model binaries default to `~/.cache/research-ultra-rag-mcp/models` and are the only cross-project shared state. `project.lock` serializes MCP and UI operations across processes so no caller observes a partial index.

## Operations
Each operation answers with the complete service payload to the workspace and the command line, and with its projection to an agent; the bullets below name what each one builds.

- `status`: read-only current/staleness inspection — the selected generation, what a prune would consider as `retained_generation_count` and `retained_generation_bytes`, the conditions a reader must act on (`blocked_by` and `degraded`, each `{check, reason, remedy}`, both omitted when empty), the per-check detail (`checks`, `not_checked`), the generated launcher state (`ui_launcher`), and the version block with the checkout it came from. A generation the app cannot serve is stated as `hybrid_ready: false` beside `generation_upgrade_required`. The command line prints the lean verdict by default and the whole payload under `status --verbose`.
- `ingest`: return the current generation for an exact no-op, advance a checkpointed build and return `in_progress`, or select a complete new generation with verified reuse; `force_recompute` bypasses reuse but may resume its own matching checkpoint. It is resumable and one build per project at a time: an operation that arrives while another process holds the project lock is rejected promptly and names the resident build's phase and progress, and it must never be queued for minutes, because the workspace is waiting on it. A checkpoint whose identity no longer matches the corpus is discarded, and the answer reports that as `superseded_build` with a message, so progress that restarts is explained instead of being mistaken for the reader's own error.
- `search`: hybrid retrieval with reranking, always; source selection (`source_ids`, `exclude_source_ids`); and the reviewed-metadata layers with any-of semantics for `projects_any`, `categories_any`, `languages_any`, `authors_any`, and `titles_any` and all-of semantics for `keywords`. Authors and titles match as case-insensitive substrings of the effective document's values, because a name is a phrase rather than a controlled tag. The engine can serve BM25 or dense alone for measurement, but neither surface exposes a mode, a view, or a freshness switch.
- `sources`: the corpus inventory, taking no parameters: inspect indexed documents and metadata, expose `discovered_sources` before ingestion, and idempotently register those stable IDs in the portable catalog so `known_sources` remains addressable after an original disappears. The registration is a durable project-state write, so the operation is not read-only even though the inventory looks like a read. It is the workspace's and the command line's answer; the agent surface serves `find_source`, which reads the same addressable set and returns only what one name matched.
- `get_passage`: retrieve neighbouring chunks from the same document.
- `set_source_inclusion`: immediately exclude or restore a reviewed source without modifying the source file; rebuild later to align the indexes. It uses the same exact-one-selector rule.
- `set_source_metadata`: save one source's reviewed metadata into the review-state JSON, rewriting only that entry so a hand edit to another entry survives. The file stays authoritative at read time, and the workspace dialog is a client of this operation rather than a second writer.

The surface has no bundle operations: `export_bundle` and `import_bundle` are not available. Reviewed metadata is written by `set_source_metadata`, and the review file stays authoritative at read time.

## Retrieval contract and recorded decisions
- A repeated passage is collapsed from the answer, and a build never drops a chunk for it. Two sources can hold the same text — an essay on its own and the same essay inside a book — and both texts are wanted, so `set_source_inclusion` does not remove the overlap: the other source still stands. A build that dropped one would decide with a comparison no caller asked for, change the generation every time it did, and leave a person with a corpus that quietly lost a passage. So both copies stay in the chunk rows, in the vector matrix and in both indexes, and each one remains independently citable: the reader who wants the essay from the second file can still ask for it by name. The search settles it instead, on the candidates that search already holds — the **words** first, which need no vector and cost nothing, then the **cosine** within `retrieval.duplicate_cosine` for the passage that says the same thing in other words. A survivor is compared only against survivors, so the better-ranked of a repeated pair is the one shown, and the pair is reported under `collapsed_repetitions` with both sources, because a person who sees one essay arriving from two files is the only one who can say whether a source should be retired. The work is `maximum_candidates`, not the square of the corpus, and the vectors are read from the generation's own portable matrix and compared as **cosines**: normalise first, or a store whose vectors are not at unit length scores two unrelated passages above any threshold this setting allows. The threshold is a setting and decides the cosine part only; words that are the same words are one passage whatever it says, a passage with no words repeats nothing, and a number no cosine can reach decides that nothing is close enough rather than switching the check off. Do not add a second similarity rule elsewhere, do not make this one advisory, and do not reintroduce a build-time pass.
- Default method: `hybrid`; diagnostic methods: `bm25` and `dense`.
- Lexical path: pinned vanilla gateway -> UltraRAG BM25.
- Dense path: FastEmbed `BAAI/bge-small-en-v1.5`, ONNX Runtime CPU, 384 dimensions, cosine distance, and a project-local index the manifest records: the exact scan by default, or the embedded Qdrant collection `research_chunks` above the documented threshold. Artifact revision: `52398278842ec682c6f32300af41344b1c0b0bb2`.
- Fusion: weighted reciprocal-rank fusion with `k=60`, BM25 weight `1.25`, and dense weight `1.0`. Do not combine raw BM25 and cosine values; their scales are unrelated.
- Candidate depth: at least 20, normally `top_k * 4`, bounded at 200 and by the current chunk count.
- `top_k` is the returned-passage budget, and the flat passage ranking is the only view: no surface and no tool can request a grouped reference view.
- Search-level source selection resolves stable `source_id` values to document IDs in the selected generation before ranking, so `top_k` is a budget inside the selection. `source_ids` includes and `exclude_source_ids` removes; both default to empty, which means include everything and exclude nothing. An include list that resolves to no document in the selected generation is an error, never a silently unfiltered search; unresolved IDs are disclosed under `filters`; and a reviewed exclusion always wins over `source_ids`.
- Keep review state hand-editable. `source-metadata.json` and `source-exclusions.json` are schema-versioned plain JSON that a person may edit directly, so the read path must keep honouring a hand edit and must reject an unknown field name, a wrong value type, or a non-normalized source path with a message naming the problem, never by ignoring it. `tests/test_review_state_edits.py` pins both halves.
- Reviewed metadata carries six independent filter layers, all authoritative at read time: `project` records which project a source was gathered for, `categories` the branch or branches it belongs to, `keywords` the terms that identify it or that it leans on, `language` what it is written in, `title` the work's title, and `authors` who wrote it as one string per name. The project, category, language, author, and title layers match any-of (`projects_any`, `categories_any`, `languages_any`, `authors_any`, `titles_any`) and the keyword layer matches all of its terms. `title` is a scalar string where the others are lists, so it is normalized on its own rather than by the list normalizer that would iterate it character by character, and authors and titles are matched as case-insensitive substrings of the effective value because a name is a phrase rather than a controlled tag. Language is the only one extraction detects, from the source's own function words against the stopword lists BM25 knows, and it records the decision in `metadata_provenance.language` (`pdf_catalog`, `epub_opf`, `text_sample`, or `missing`); detection answers nothing rather than guessing, and a code BM25 cannot tokenize is a valid value because the field describes the source rather than the index. Filtering resolves the current reviewed overlay to document IDs at query time and must never bake metadata into an index. `status.categories`, `status.projects`, and `status.languages` are the inventories (value plus searchable source count) of the selected generation, with reviewed exclusions removed. Because one process serves one project, the project layer is normally a passthrough inside an app and earns its place when a corpus is bundled, imported, or shared.
- Chunking: UltraRAG token chunker with the GPT-2 tiktoken encoding, default and maximum 384 tokens, overlap 64. The cap stays below the embedding model's 512-token input limit despite tokenizer differences; do not raise it without an explicit long-input strategy and tests.
- Reranking: FastEmbed `Xenova/ms-marco-MiniLM-L-6-v2`, CPU, lazily loaded, applied to at most 50 candidates. Artifact revision: `a09144355adeed5f58c8ed011d209bf8ee5a1fec`. It is **always on** for the operation because it is the largest measured quality gain (`MEASUREMENTS.md`); an unavailable model must degrade to the unranked candidate order with `rerank_fallback` in the response, never fail the search.
- A dense index owns only vectors and identifiers (`chunk_id`, `document_id`, `source_id`). `chunks.jsonl` remains the canonical passage and locator store; document metadata and provenance remain canonical in the manifest, so filtering resolves current metadata to document IDs at query time.
- Contextual chunk headers live in a chunk's `embedding_text`, and the canonical `contents` is what BM25 indexes, what a search returns, and what the artifact lookup hashes to resolve a passage back to its chunk. A header therefore covers a vector and nothing else, and vector reuse is keyed on `contents`, so a generation built with headers recomputes its vectors rather than reusing them.
- The generation-local SQLite artifact lookup contains only identifiers, ordinals, content hashes, byte offsets into canonical chunk/unit JSONL, and one integer retrieval verdict per chunk. It must never duplicate passage or extraction text. Use it for candidate retrieval, neighboring passages, document-scoped reuse, and exact-text vector reuse; reconstruct it from canonical artifacts when a legacy generation or portable bundle does not contain it.
- Precompute a retrieval verdict when the lookup is built and reject candidates from that stored value rather than rescanning text per query. The verdict is a bitmask over properties of the chunk alone (`chunk_health_flags`: corrupt text, extraction artifact), never a property of the query, and it must mirror the query-time check exactly so rejection counters and withheld disclosures cannot change. Keep a fallback that recomputes it from text when a lookup predates the column, and recompute reason codes only for a chunk the verdict flags as corrupt, because the response discloses them.
- Qdrant is used instead of FAISS here because payload filtering and scored results are needed. FAISS remains an upstream vanilla capability. Milvus is intentionally not required because this app targets local project use.
- Category and keyword lists use AND semantics; document IDs use membership semantics. Resolve current reviewed filters to matching document IDs, use those IDs for dense retrieval, and verify results against canonical documents rather than copying mutable metadata into a dense index.
- Raw BM25 scores are unavailable from the pinned UltraRAG tool. Report its rank, never synthesize a score. Dense/fusion/reranker scores are ranking signals, not calibrated confidence or truth probabilities.
- BM25 candidates require at least one query token outside the corpus language's function-word set. English uses bm25s's fuller list, which also stops question and do-support words, so a query that carries no topic abstains instead of matching on a function word for a topic. Dense candidates require cosine similarity `>= 0.72`, and a candidate below that floor is admitted only when the query's best candidate cleared the floor and this one is within `retrieval.dense_relative_similarity_margin` of it. The rescue must never manufacture support: a query nothing cleared the floor for still abstains. A candidate also needs `retrieval.minimum_passage_words` words and `retrieval.minimum_passage_token_fraction` of the generation's recorded `chunking.size` in tokens before it counts as evidence, because chunks never span extraction units and a short unit therefore becomes a chunk that matches a query about its own words: an index line, a heading, a caption, or a copyright line. Count the token floor over the returned text with the generation's recorded tokenizer, so the unit matches the chunker and a contextual header cannot enlarge a fragment. Report the floor, the margin, both length rules, the query's best similarity, the admitted and rejected counts, and the sources of the candidates each rule dropped, in the payload. Reject extraction artifacts before the optional reranker and permit fewer than `top_k`, including zero.
- Final chunk artifacts use one cleaned semantic-content field, `contents`; `text` and `embedding_text` are read-only legacy compatibility fallbacks. Never inject paths, authors, citations, or repeated document titles into each indexed passage. Public results project `contents` as `text`.
- Normalize layout wrapping before chunking, remove controls/soft hyphens, and join alphabetic line-end hyphen splits. Preserve all other wording and punctuation, and keep `direct_quote_safe=false` on every passage the service builds; the payload carries that field, while a answer states the rule once rather than repeating it per passage.
- Do not retain raw coordinate extraction. The untouched PDF/EPUB is the quote authority. Preserve legend-marker meaning in cleaned-unit annotations.
- Schema-1 generations are BM25-only. Keep them usable when the reader explicitly requests `bm25`; require a new ingestion before dense or hybrid search.
- Exclusions are path-based, stored outside generations, and applied to BM25, dense, source-list, and passage results immediately. Ingestion snapshots the exclusion revision and omits excluded documents from both indexes. Inclusion can only restore current retrieval immediately if the current generation still contains that source.
- Project portability is copying, not a format: `sources/` plus `.research-rag/` is the whole project, and `runtime/` rebuilds. A relocated runtime root stays claimed by its owning project marker, so never point a second project at one.

## Metadata and extraction contract
- Resolve each field independently. Automatic PDF precedence is valid visible front matter, valid embedded metadata, then the filename stem for title only. Automatic EPUB precedence is valid OPF metadata, visible title/byline, then the filename stem for title only. Apply reviewed metadata afterward as the authoritative read-time per-field overlay.
- Keep automatic metadata rules generic and conservative. Never add a source-, title-, author-, or publisher-specific extraction exception to fix one document. Expose uncertainty through provenance and warnings, then use a reviewed `set_source_metadata` override for the exceptional document.
- Never infer authors from filenames. Reject DOI/URL/export-junk titles, move a detected DOI to its own field, and expose per-field provenance plus concrete review warnings. Provenance and those inspectable warnings are the complete uncertainty model.
- Apply the deterministic English-oriented text-health classifier to complete extraction units and automatically extracted titles/authors. Retain only locator/reason diagnostics for rejected units, never guessed repairs or their garbage text. Reviewed metadata remains authoritative. Apply the same guard at retrieval time for older generations.
- Withhold text only for corruption evidence: replacement characters, private-use or unassigned code points, or a known damaged encoding sequence. A single replacement character counts only with corroborating corruption evidence. Script mixing and non-Latin dominance never withhold a unit, a chunk, or a passage, because English-language scholarship legitimately quotes other scripts; the advisory script note belongs to the payload rather than to every returned passage.
- Fold only formula-font letters (Mathematical Alphanumeric Symbols) and the alphabetic presentation ligatures (fi, fl, ff) so a typed query can match the printed text. Leave every other character canonical: do not apply global NFKC, because it would also fold superscripts, subscripts, and symbols that carry meaning in citations and notation.
- Audit every built chunk against the embedding model's token limit, record the count and a truncation flag on the chunk record, and report the aggregate in build metrics and, in the payload, per passage. When the tokenizer cannot be inspected, record the audit as unavailable rather than failing the build or inventing a count.
- Treat the embedding inference batch size as a padding decision, not a throughput dial: FastEmbed pads every sequence to the longest member of its batch, so a larger batch makes short chunks pay for the longest one. Keep `EMBEDDING_INFERENCE_BATCH_SIZE` at 1 unless a measurement on the target corpus says otherwise, and record any change in `MEASUREMENTS.md`. Do not raise it "to go faster" without measuring padded tokens, and keep `--embedding-threads` unset by default because its optimum is machine-specific.
- Disclose withholding instead of hiding it. Report reason codes, counts, and example chunk IDs in the full-detail search payload, and record corpus-level withheld counts and reasons in the generation build metrics that the full-detail `status` returns.
- Keep checkpoints durable at the finest practical granularity: one extracted document, one PDF scan batch, one extraction-unit batch, one embedding batch, and one index batch. Reduce the cost of each durable write rather than widening the resume granularity, so a crash never redoes more than one bounded batch. Each phase's batch size is a documented constant — `PDF_PAGE_BATCH_SIZE`, `CHUNK_BATCH_UNITS`, `EMBEDDING_BATCH_SIZE` — and it must stay bounded and small enough that redoing one batch is cheap. Never replace a bounded batch with a whole-phase commit.
- Exclude every nonempty chunk that contains no Unicode alphanumeric content, both while ingesting and at retrieval time for older generations. Text, numbers, and formulas containing at least one letter or digit are not classified as symbol-only; the other extraction-artifact rules still apply.
- Treat the portable reviewed-metadata file as an authoritative read-time overlay on the selected immutable generation. Apply it consistently to source listings, search results, reference groups, citations, neighboring passages, and category/keyword filters without rewriting generation files.
- Do not mark a selected generation stale merely because its metadata snapshot differs from the current reviewed overlay. Report that the overlay is active. A source absent from the selected generation still requires ingestion before any of its metadata can appear in retrieval.
- Do not copy mutable category or keyword values into a dense index. Translate current reviewed filters through selected-generation document IDs and verify them against the overlaid canonical documents and chunk records.
- Omitted fields remove their prior reviewed overrides. If an old generation cannot recover automatic bibliography hidden by a removed override, prefer a safe missing/filename fallback with an explicit warning; a later ingestion may recover automatic metadata from the original.
- Inspect the first five text-bearing PDF pages for identity. Keep physical page and available page-label locators.
- For EPUBs, retain the spine section identity and a deterministic XHTML block position for every semantic unit. Preserve an existing element ID or named anchor as an exact fragment when available; never label a synthetic block path as an EPUB CFI or imply quotation-level precision.
- Use coordinate blocks to restore column order, remove repeated margins/page numbers, and distinguish prose, lists, tables, and figures. Do not infer visual relationships not expressed by captions, legends, or labels.
- Hash all discovered sources before ingestion decides whether work is needed. Exact source bytes, source-exclusion decisions, chunk settings, processing policies, and model fingerprints are required for a no-op. Reviewed metadata is excluded from build/checkpoint identity because it is a read-time overlay.
- Reuse extraction units/chunks only for a source with matching bytes, automatic-metadata storage policy, and processing fingerprints. Generation artifacts retain automatic bibliography beneath the reviewed overlay. Reuse a vector only when canonical `contents`, model revision, and dimension match exactly. Legacy text fields may be read only to upgrade an older generation.
- Always reconstruct complete BM25 and dense indexes for a changed generation; never update selected indexes in place. `force_recompute=True` disables all document, chunk, and vector reuse.
- Check the soft work budget only between atomic units: source hashes, fixed eight-page PDF scan/extraction batches, EPUB spine sections, extraction-unit chunking, 64-text embedding batches, and 64-point dense uploads. Treat BM25 finalization as one restartable unit and re-hash all sources before activation.

Do not move the dense backend implementations into the vanilla gateway or patch UltraRAG for this feature. The research-specific integration deliberately lives in this repository so vanilla can continue tracking upstream safely.

## Working rule
Commit and push after every change, without waiting to be asked. Run the validation commands below, commit the change with a message that says what it changes and why, and push to `origin main` before starting anything else. A change that is not on the remote is a change that is lost, and the collection cannot record a pointer to a commit that is not there.

## Safe change rules
- Use `pathlib.Path`, type hints, and JSON-serializable payloads.
- Answer through the core, never around it. The workspace and the agent surface call a `ResearchService` method on the app's own service; the command line reaches that same service through the control API, and builds one in process only when the project has no app up. One capability has one implementation and the three surfaces cannot disagree.
- Keep the package split at `surfaces/`. The engine imports neither Starlette nor FastMCP, no engine module imports a surface, and no surface imports another, so the boundary is a directory rather than a list of filenames someone has to remember.
- Keep configuration resolution independent of work no operation has asked for. The app resolves its settings and starts serving before it imports the retrieval stack or opens the gateway, so a project reads fine on a machine where the UltraRAG runtime is not installed yet. A gateway that cannot start is reported by the operation that needed it, never by a surface that never appeared, and the answer names the log the reason was written to. `tests/test_integration.py` pins this with a gateway that exits immediately.
- Keep blocking extraction and filesystem scans outside the event loop.
- Set process-level resource policy once, at start: `runtime.nice` is applied by the command line and by the workspace before anything is spawned, so children inherit it and one setting reaches the whole process tree without a second copy of the logic.
- Signal only a process this project can prove it owns. The stop sweep requires an app entry point in the command line *and* this project as `--project-root`, so it cannot touch a shell, an editor, another project's process, or a process of the other product in this collection, and it never signals the process doing the sweep.
- Serialize every project operation with both the in-process service lock and the cross-process `project.lock`.
- Prefer new immutable generations to in-place index mutation.
- Validate a new artifact before updating a pointer to it.
- Keep model downloads lazy: the embedding model is needed during ingestion; the reranker model only when a search reranks. In offline mode, require an existing shared cache, with read-only fallback to an existing legacy project-local cache. Never migrate or delete that legacy cache automatically.
- Do not expose the underlying vanilla operations.
- Do not silently skip a selected PDF/EPUB that fails extraction; fail the new generation and leave the previous current generation intact.
- Empty, symbol-only, or corrupt upstream chunk records may be discarded when their source still has at least one searchable chunk; record each discarded count separately. Fail the build when filtering leaves a source with none.
- Retain explicit limitations when a feature is not implemented.
- Keep the personal-material block in `.gitignore` accurate: the author's own research projects, notes, drafts, and every original PDF/EPUB are local-only and must never be committed, published, or pushed from this repository. Add new personal locations to that block rather than ignoring them silently.

## Validation
Before finalizing a change, run:

```bash
uv lock --check
uv run ruff format --check .
uv run ruff check .
uv run pytest -q
uv run python -m compileall -q src tests
```

For a retrieval-quality claim, run `uv run python scripts/evaluate_retrieval.py --project <project> --offline` (add `--validate-only` to check the judged set first) and record the result in `MEASUREMENTS.md`. Never change a documented retrieval default from an unrecorded run or a single query.

Workspace adapter changes must cover safe source-file resolution, the arguments each operation forwards, and the real host against an existing project without mutating its sources. Shared workspace, JSON validation, capability, and same-origin changes belong in `ui-ultra-rag-mcp` and must pass that package's own tests before updating the pinned commit here.

For source or retrieval changes, the integration test must still launch the real vanilla gateway, build BM25 and dense indexes, run hybrid and dense search, retrieve the known passage, and prove that a neighboring Markdown file was excluded. It must then restart offline and repeat hybrid reranked search from the caches. Unit tests must cover RRF and failure atomicity without depending on model downloads. They must also cover no-op ingestion, additions, changes, removals, reviewed metadata/exclusions, same-size/same-mtime byte changes, forced regeneration, exact-text vector reuse, and final artifacts. They must also prove that post-ingestion metadata corrections immediately affect every read surface and filter mode without modifying generation files.

For significant extraction changes, also test a representative real collection without writing into its source directory.
