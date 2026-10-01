# AGENTS.md

The engineering guide for AI coding agents working in `research-rag`.

## Objective

Serve a project-scoped research knowledge base built from original PDF and EPUB sources, over one loopback port, from one process. A person searches their corpus, reads each passage beside its source, locator, and reviewed bibliography, and records what they decide about each source. An agent retrieves the same passages through the same service. The app never writes an answer: the reader concludes, and the untouched PDF or EPUB is the quote authority.

The workspace, the agent's tools, and the command line are three front ends to one process that owns the project, its lock, and its UltraRAG gateway. The only question a reader has to ask is whether the app is up.

Never add research behaviour to `vanilla-ultra-rag-mcp-server` or `ui-ultra-rag-mcp`; both are separate, versioned projects.

## Documentation responsibilities

Each fact has one home. The other files point at it and never repeat it.

| File | Owns |
|---|---|
| `README.md` | the user manual: what it does, how to run it, what it cannot do. Every installed command. |
| `STORAGE.md` | the state format: every file, every field, portability, hand edits. The only place field names are defined. |
| `FEATURES.md` | the capability inventory: what comes from UltraRAG, what this app adds, what is excluded, what is planned. |
| `MEASUREMENTS.md` | the numbers, the limits they do not establish, and every constant, model, revision, and threshold. No other file states one. |
| `ROADMAP.md` / `TODO.md` | deferred ideas / open work. |
| `AGENTS.md` | this file: rules that are not derivable from the code or the tests. |
| A module's own docstring | why that module does what it does, and what it may never do. |

The last row is the load-bearing one. A rule stated here and again in the module it governs is a rule in two places, and the two will disagree. `tests/test_documentation.py` fails when a command is registered and not in the `help` menu, or listed twice, and `tests/test_review_state_edits.py` pins both halves of the hand-edit rule.

### Markdown describes the present

- Git is the archive. No markdown file may record how the software got here: no review logs, finding lists, change histories, before/after narratives, "Step N" sequences, or **recorded decisions**. A decision that still binds is a rule, stated once, as a rule. A decision that no longer binds is deleted, not filed.
- No "previously", "used to", "was", "before this change", or "Step N". State the current fact and stop.
- A finished item leaves `TODO.md` and appears nowhere else. A superseded number leaves `MEASUREMENTS.md`.
- Keep numbers current. Re-measure when the corpus, an extraction policy, or a retrieval default changes, and update every quoted figure rather than leaving one stale value contradicting the rest.
- Cross-references name a file by path or a harness by name, never a section number that can drift.
- `README.md` must not compare or link to sibling projects in this collection.

### Only direct, concise language

- One sentence per fact. No preamble, no hedging, no restating what the reader just read.
- Cut filler: *very*, *really*, *actually*, *simply*, *just*, *it is worth noting*, *importantly*, *of course*.
- Do not sell. No superlatives or dramatic framing; a number, a mechanism, or a limit is the argument.
- Prefer a concrete noun and an active verb.
- Length is not thoroughness. Split long sentences, turn paragraphs into bullets, never open an item by restating its own title, and delete any sentence whose removal changes nothing.

## Presenting decisions to the user

Any question needing a user choice must be a numbered list of concrete options, never an open question. Each option carries a short identifier (`A`, `B`, `C`, …), a one-line statement of what it does, the smallest change it requires, pros and cons covering cost, risk, and the effect on the research contract, and whether it is reversible. Mark exactly one as the recommendation and say why in one sentence. Include a "no change" option whenever work can proceed without an answer.

Never implement a choice that changes generation artifacts, retrievable evidence, the portable-state contract, or on-disk layouts before the user has chosen it.

## Current baseline

- Package `research-rag` 0.1.0, Python `>=3.11,<3.13`. This repository's own code is Apache-2.0 (`LICENSE`); `NOTICE` records the upstream, model, retrieval-component, and AGPL-3.0 extraction terms, which stay separate from that grant.
- Command: `research-rag`, with `init`, `status`, `ingest`, `search`, `sources`, `passage`, `include`, `exclude`, `metadata`, `config`, `doctor`, `start`, `ui`, `clients`, `disconnect`, `mcp`, `serve`, `stop`, `generations`, `remove-generation`, and `help`. `help` prints the commands grouped by the work plus a page per subject, needs no project, and delegates a command name to that command's own usage.
- Agent surface: seven tools and one resource at `<app>/mcp` on the app's own port, plus the stdio bridge.
- Pinned: vanilla gateway `fc339c259a672ca4dacb525840eba851d01c4b75`, shared UI `924a281`, shared settings-core `cbd47bb46efe85340de34f8f8f13fc6516e7fecb`, UltraRAG `0.3.0.2` at `3a709a2aea3fbe46acca59c422621c94b6e86857`, `bm25s` fork `20f6c02`.
- On-disk state is byte-compatible with `research-ultra-rag-mcp`, which is frozen and still installed on the machines that carry it, so both keep reading and writing the same projects. `USER_CONFIG_DIRECTORY`, `SETTINGS_ENVIRONMENT_PREFIX`, the model-cache directory, and `runtime.tool_detail` keep that product's names; `tests/test_data_roots.py` states each one. This app is the only one of the two that changes.
- `bm25s` is forked for a one-line non-ASCII stopword fix; `settings` fails fast on a list that cannot round-trip through it. Revert the pin only once upstream fixes it.
- `pymupdf` raises `IndexError` from `Page.get_label()` when a document's page-label tree starts after the page asked about, failing a whole ingestion. Upstream issue 5140; `_pdf_locator` catches it and falls back to the physical page. Narrow the guard when that closes.
- `FEATURES.md` section 1.1 must state the reason for every UltraRAG capability this app does not use, and what would revisit the decision.

## Rules

Rules that a module's own docstring or a test already carries are not repeated here. What follows is what a reader of this file cannot get anywhere else.

### The contract a reader relies on

- Retain the prominent UltraRAG acknowledgement in `README.md`, the root `NOTICE`, upstream project links, licence information, and the independent-project disclaimer. Credit THUNLP, NEUIR, OpenBMB, AI9stars, and the upstream contributors using the wording UltraRAG's own README supports. Do not imply endorsement.
- Never edit or write an original source document. `sources/` is the authority for a quotation.
- `text` in a result is cleaned semantic text, never a transcript. `direct_quote_safe` is `false` on every passage the app builds, and that flag is the only place the rule appears per passage.
- An agent's tools answer one question at a time, and no tool lists the corpus. The workspace and `sources` hold the inventory. Never let a tool answer scale with the size of the corpus. A generation the app cannot serve is stated as `hybrid_ready: false` beside `generation_upgrade_required`, and the old one stays searchable with a warning.
- Every operation takes only the arguments its reader must decide. The agent's tools offer no retrieval mode, no output view, and no chunk tuning, because a capability the measurements already answer is an engine setting. The one recorded exception is `search --method` and `--no-rerank`, which exist to reproduce a row of `MEASUREMENTS.md`.
- `status` and an agent's `status` are the same answer. Two readers are bounded and share `tool_views`, so they cannot disagree.
- Generation of an answer is out of scope. Adding it would be a deliberate research feature with its own API, citation contract, and tests, not a patch. There are no bundle operations: `export_bundle` and `import_bundle` are not available.

### The state a reader must be able to trust

- One app process per configured project root, and every surface shares that process, its project lock, and its single UltraRAG gateway. The app is up or it is down; a command reaches the running app over the control API, and a project with no app up is answered in process, because a read of local state must not refuse itself because no daemon is running.
- Every project-owned artifact lives beneath `<project>/.research-rag`, apart from the machine-local launcher symlink. `STORAGE.md` has the layout, the runtime-root rules, and the marker.
- Every path this app names inside a project carries `research-rag`, including the launcher's pid, port, log, and lock files. Two products sharing a pid file would let each stop the other's process.
- Share only immutable model binaries through the configured user cache. Never place documents, metadata, chunks, vectors, indexes, logs, or query state in global storage.
- Preserve deterministic source IDs, document IDs, chunk IDs, source paths, and locators. A `source_id` follows the normalized source-relative path and survives a byte change; a `document_id` identifies a path and a content version. Chunk IDs may change when content or chunking changes. Never imply document or chunk IDs are permanent across incompatible generations.
- Switch `current.json` only when a generation is completely indexed, and validate an artifact before pointing at it. Keep BM25 and dense in the same generation, and never select a generation unless both validate.
- Keep review state hand-editable. A hand edit must be honoured, and an unknown field, a wrong type, or a non-normalized path must be refused with a message naming the problem, never ignored.
- Reviewed metadata is a read-time overlay on the selected generation. Never write it into a generation file, and never mark a generation stale because the overlay moved.
- Exclusions are explicit, reversible, project-local, and enforced by every retrieval surface at once. Do not add automatic duplicate guessing.
- The on-disk contract is frozen while `research-ultra-rag-mcp` is installed: a change here that it cannot parse is a reader answering with state it could not have written. `project.json` is the only file it reads a known subset of.
- `registry.py` is a pointer file — an id, a name, and a root per project. Never let it grow into a cache of a project's state, and never let one project's record decide what another may read.
- The layer machinery is not this repository's; it lives in `config-ultra-rag-mcp`. This repository owns `SETTINGS`, `EffectiveSettings`, the packaged `default.toml`, and the three names it resolves its own layers by.
- Keep schema and policy versions, the retrieval-method set, and the boundary names in code, so a settings file cannot forge what a generation is or where the app may write. A tunable that decides what a generation contains enters the retrieval-policy fingerprint or the recorded chunk settings; one that cannot is a runtime setting.
- `use_generation` and `remove_generation` match a caller-supplied name against the pattern the builder writes before it becomes a path, and both take the project lock. Both refuse the generation search reads, because nothing refuses a delete loudly: an unlinked memory mapping keeps reading the old bytes and a later open returns a missing-file error. `remove_generation` also refuses a generation a pending activation has named, and requires the id twice.
- The durable-write order is required and expected: group directory fsyncs inside a unit, and never let a commit point become durable before the artifacts it describes. A file handed off to a peer process and rewritten before every use may skip fsync.
- A checkpoint is durable per bounded batch, not per phase. Redo one batch, never a phase.
- `ingest` is one build per project at a time. An operation arriving while another process holds the project lock is rejected promptly and names the resident build's phase; it must never be queued for minutes, because the workspace is waiting on it.
- Reuse extraction and vectors only for a source with matching bytes, storage policy, and processing fingerprints, and a vector only when the canonical text, model revision, and dimension match exactly. `force_recompute=True` disables all reuse.
- Record the dense backend in the manifest and dispatch from that record. Never rebuild a generation with a different backend.
- Do not silently skip a selected PDF/EPUB that fails extraction. Fail the new generation and leave the current one intact.

### Boundaries a contributor must not cross

- Keep the package split at `surfaces/`. The engine imports neither Starlette nor FastMCP, no engine module imports a surface, and no surface imports another. `tests/test_architecture.py` enforces it.
- Neither surface may re-declare the other's tool, resource, or operation. A second copy of an operation is a second place for it to be wrong.
- The workspace's profile turns off what it does not serve, and `ResearchUIAdapter._arguments` drops every argument an operation does not accept. Never let a workspace control travel as an argument the app ignores.
- The shared UI is pinned by commit. Keep the adapter, source authorization, and the process host here; do not copy the shared static workspace into this package.
- Do not let workspace code read or mutate generation artifacts directly. It calls `ResearchService`, except for safely serving an allowlisted original from the source root.
- Keep browser and control write endpoints same-origin, JSON-only, and loopback-only.
- Never accept an arbitrary filesystem output path. Ingestion selects only regular PDF and EPUB files beneath the configured sources directory. Reject source symlinks and path traversal.
- Never expose the underlying vanilla operations, and never patch UltraRAG or move the dense backends into the gateway.
- Answer through the core. The workspace and the agent surface call a `ResearchService` method; the command line reaches that same service over the control API and builds one in process only when the project has no app up. One capability has one implementation, and the three surfaces cannot disagree.
- Keep the workspace bound to loopback. Do not add remote exposure or authentication assumptions without an explicit security design.
- Never hand a child process `dict(os.environ)`; use `child_process_environment()`. Never point this app at a second copy of itself.
- Signal only a process this project can prove it owns. The stop sweep requires this app's entry point in the command line *and* this project as `--project-root`, and never signals the process doing the sweep.
- Keep the author's own projects, notes, drafts, and every original PDF/EPUB out of git. Add new personal locations to the block in `.gitignore`.
- Retain explicit limitations when a feature is not implemented. Never present planned work as shipped.

### Reporting a condition

- Name every condition a reader must act on, and put the remedy beside it. `health.py` owns one check per condition in one of four states: `ok`, `warn` (the answer is worse), `blocked` (the app cannot work), and `unknown` (the check did not run, which never reads as healthy).
- Every check reads: no process, no network, no write, so the report can be built before the gateway is opened. A condition the report cannot state must read as unknown rather than be omitted.
- `blocked_by` and `degraded` are omitted when empty and both present together when a project is healthy. A reader who sees one must be able to name the fix without reading this file, so a reason is a sentence and a remedy is a pasteable command. The per-check detail is a `doctor` reader.
- `doctor` reads the same report, so the two surfaces cannot disagree. It writes nothing at all, and `--prefetch-models` and `--repair-runtime` are the only operations that reach the network. A repair that discards evidence moves it aside.
- A failure message must name the reason and where to read the rest, so `Connection closed` is never the whole of what a reader is told. No process-tree inspection: the vanilla tool name says which component was busy, and that component writes its own log.
- A gateway that cannot start is reported by the operation that needed it, never by a surface that never appeared. The app resolves its settings and starts serving before it imports the retrieval stack or opens the gateway, so a project reads fine where the runtime is not installed. `tests/test_integration.py` pins this.
- Keep blocking extraction and filesystem scans outside the event loop.
- Serialize every project operation with both the in-process lock and the cross-process `project.lock`.

## Architecture

```text
MCP client ── HTTP /mcp ─┐
                       │
stdio client ── bridge ──┤
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

The vanilla gateway is an implementation dependency below the app, not a second surface. The shared workspace package owns no research state, and the stdio bridge is a transport, not a second copy of the tools.

## Working rule

Use `pathlib.Path`, type hints, and JSON-serializable payloads. Change files under `evaluation/` only with the protocol that file describes.

Commit and push after every change, without waiting to be asked. Run the validation below, commit with a message that says what the change does and why, and push to `origin main` before starting anything else. A change that is not on the remote is lost, and the collection cannot record a pointer to a commit that is not there.

## Validation

```bash
uv lock --check
uv run ruff format --check .
uv run ruff check .
uv run pytest -q
uv run python -m compileall -q src tests
```

For a retrieval-quality claim, run `uv run python scripts/evaluate_retrieval.py --project <project> --offline` (add `--validate-only` first) and record the result in `MEASUREMENTS.md`. Never change a documented retrieval default from an unrecorded run.

Workspace adapter changes must cover safe source-file resolution, the arguments each operation forwards, and the real host against an existing project without mutating its sources. Shared workspace, JSON validation, capability, and same-origin changes belong in `ui-ultra-rag-mcp` and must pass that package's own tests before updating the pinned commit here.

For source or retrieval changes, the integration test must still launch the real vanilla gateway, build both indexes, run hybrid and dense search, retrieve the known passage, prove a neighbouring file was excluded, then restart offline and repeat hybrid reranked search from the caches. Unit tests must cover RRF and failure atomicity without model downloads, plus no-op ingestion, additions, changes, removals, reviewed metadata and exclusions, byte changes that preserve size and mtime, forced regeneration, vector reuse, and final artifacts. They must also prove that post-ingestion metadata corrections immediately affect every read surface and filter mode without modifying generation files. For significant extraction changes, also test a representative real collection without writing into its source directory.
