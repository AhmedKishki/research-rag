# AGENTS.md

- This file is the engineering guide for AI coding agents working in `research-rag`.

## Objective

- The app serves a project-scoped research knowledge base built from original PDF and EPUB sources, over one loopback port, from one process.
- A person searches their corpus, reads each passage beside its source, locator, and reviewed bibliography, and records what they decide about each source.
- An agent retrieves the same passages through the same service.
- The app never writes an answer: the reader concludes, and the untouched PDF or EPUB is the quote authority.
- The workspace, the agent's tools, and the command line are three front ends to one process that owns the project, its lock, and its UltraRAG gateway.
- The only question a reader has to ask is whether the app is up.
- No research behaviour belongs in `vanilla-ultra-rag-mcp-server` or `ui-ultra-rag-mcp`, which are separate, versioned projects.

## Documentation responsibilities

- Each fact has one home.
- Every other file points at that home and never repeats the fact.

| File | Owns |
|---|---|
| `README.md` | the user manual: what it does, how to run it, what it cannot do. Every installed command. |
| `STORAGE.md` | the state format: every file, every field, portability, hand edits. The only place field names are defined. |
| `FEATURES.md` | the capability inventory: what comes from UltraRAG, what this app adds, what is excluded, what is planned. |
| `MEASUREMENTS.md` | the numbers, the limits they do not establish, and every constant, model, revision, and threshold. No other file states one. |
| `ROADMAP.md` / `TODO.md` | deferred ideas / open work. |
| `AGENTS.md` | this file: rules that are not derivable from the code or the tests. |
| A module's own docstring | why that module does what it does, and what it may never do. |

- The last row of that table is the load-bearing one.
- A rule stated here and again in the module it governs is a rule in two places, and the two will disagree.
- `tests/test_documentation.py` fails when a command is registered and not in the `help` menu, or listed twice.
- `tests/test_review_state_edits.py` pins both halves of the hand-edit rule.

### Markdown describes the present

- Git is the archive.
- No markdown file records how the software got here: no review logs, finding lists, change histories, before/after narratives, "Step N" sequences, and no recorded decisions.
- A decision that still binds is a rule, stated once, as a rule.
- A decision that no longer binds is deleted rather than filed.
- No document says "previously", "used to", "was", "before this change", or "Step N".
- A document states the current fact and stops.
- A finished item leaves `TODO.md` and appears nowhere else.
- A superseded number leaves `MEASUREMENTS.md`.
- A changed corpus, extraction policy, or retrieval default requires a re-measurement.
- A re-measurement updates every quoted figure rather than leaving one stale value contradicting the rest.
- A cross-reference names a file by path or a harness by name, never a section number that can drift.
- `README.md` never compares or links to sibling projects in this collection.

### Only direct, concise language

- One sentence carries one fact.
- No sentence opens with a preamble, a hedge, or a restatement of what the reader just read.
- Cut the filler words *very*, *really*, *actually*, *simply*, *just*, *it is worth noting*, *importantly*, and *of course*.
- Do not sell.
- No superlative and no dramatic framing is allowed: a number, a mechanism, or a limit is the argument.
- Prefer a concrete noun and an active verb.
- Length is not thoroughness.
- A long sentence splits, a paragraph becomes bullets, and an item never opens by restating its own title.
- Any sentence whose removal changes nothing is deleted.

## Presenting decisions to the user

- Any question needing a user choice is a numbered list of concrete options, never an open question.
- Each option carries a short identifier (`A`, `B`, `C`, …) and a one-line statement of what it does.
- Each option carries the smallest change it requires.
- Each option carries pros and cons covering cost, risk, and the effect on the research contract.
- Each option states whether it is reversible.
- Exactly one option is marked as the recommendation, and one sentence says why.
- A "no change" option is included whenever work can proceed without an answer.
- No choice that changes generation artifacts, retrievable evidence, the portable-state contract, or on-disk layouts is implemented before the user has chosen it.

## Current baseline

- The package is `research-rag` 1.0.0 and the Python range is `>=3.11,<3.13`.
- That version is the release version, which `pyproject.toml` declares.
- The installed distribution's metadata is where the app reads its own version.
- `update` compares that version with the release the remote publishes rather than with the branch head.
- `tests/test_release.py` holds the check that the declared version is one a release tag may carry.
- `tests/test_release.py` holds the check that a local tag carrying that version points at the commit this checkout is on.
- This repository's own code is Apache-2.0, and `LICENSE` carries that text.
- `NOTICE` records the upstream, model, retrieval-component, and AGPL-3.0 extraction terms, which stay separate from that grant.
- The one command is `research-rag`.
- `research-rag` registers `init`, `status`, `ingest`, `search`, `sources`, `passage`, `include`, `exclude`, `metadata`, `config`, `doctor`, `start`, `ui`, `clients`, `disconnect`, `mcp`, `serve`, `stop`, `generations`, `remove-generation`, `install`, `update`, and `help`.
- `install` puts the command on the account's `PATH` and writes desktop menu entries, and it needs no project.
- `update` reports what a newer version means and applies it, and it needs no project.
- No command at all opens the workspace in a browser and serves it from that terminal, asking which project when there is more than one.
- A project that is already served is reported rather than started a second time.
- `help` prints the commands grouped by the work plus a page per subject, needs no project, and delegates a command name to that command's own usage.
- `--version` prints the app version, the installed version, the shared workspace version, and whether a restart is required.
- `update` reports the same four numbers from the same functions.
- The agent surface is eight tools and one resource at `<app>/mcp` on the app's own port, plus the stdio bridge.
- `mcp` takes `--project-name`.
- `RESEARCH_ULTRARAG_PROJECT_NAME` names the same thing for a client that can pass only an environment.
- The pinned vanilla gateway is `fc339c259a672ca4dacb525840eba851d01c4b75`.
- The pinned shared UI is `1db180e`.
- The pinned shared settings-core is `cbd47bb46efe85340de34f8f8f13fc6516e7fecb`.
- The pinned UltraRAG is `0.3.0.2` at `3a709a2aea3fbe46acca59c422621c94b6e86857`.
- The pinned `bm25s` fork is `20f6c02`.
- The on-disk state is byte-compatible with `research-ultra-rag-mcp`, which is frozen and still installed on the machines that carry it.
- Both products keep reading and writing the same projects for that reason.
- `USER_CONFIG_DIRECTORY` keeps the frozen product's name.
- `SETTINGS_ENVIRONMENT_PREFIX` keeps the frozen product's name.
- The model-cache directory keeps the frozen product's name.
- `runtime.tool_detail` keeps the frozen product's name.
- `tests/test_data_roots.py` states each of those four names.
- This app is the only one of the two that changes.
- `bm25s` is forked for a one-line non-ASCII stopword fix, and `settings` fails fast on a list that cannot round-trip through it.
- The `bm25s` pin is reverted only once upstream fixes that.
- `pymupdf` raises `IndexError` from `Page.get_label()` when a document's page-label tree starts after the page asked about, and that fails a whole ingestion.
- `_pdf_locator` catches that `IndexError` and falls back to the physical page.
- The `_pdf_locator` guard narrows when upstream issue 5140 closes.
- `FEATURES.md` states the reason for every UltraRAG capability this app does not use, and what would revisit each decision.

## Rules

- A rule that a module's own docstring or a test already carries is not repeated here.
- What follows is what a reader of this file cannot get anywhere else.

### The contract a reader relies on

- The prominent UltraRAG acknowledgement survives in `README.md`, the root `NOTICE`, upstream project links, licence information, and the independent-project disclaimer.
- The credit names THUNLP, NEUIR, OpenBMB, AI9stars, and the upstream contributors, using the wording UltraRAG's own README supports.
- The credit never implies endorsement.
- No original source document is edited or written.
- `sources/` is the authority for a quotation.
- `text` in a result is cleaned semantic text and never a transcript.
- `direct_quote_safe` is `false` on every passage the app builds, and that flag is the only place the rule appears per passage.
- An agent's tools answer one question at a time, and no tool lists the corpus.
- The workspace and `sources` hold the inventory.
- A tool answer never scales with the size of the corpus.
- A generation the app cannot serve is stated as `hybrid_ready: false` beside `generation_upgrade_required`.
- A generation the app cannot serve leaves the old one searchable with a warning.
- Every operation takes only the arguments its reader must decide.
- The agent's tools offer no retrieval mode, no output view, and no chunk tuning, because a capability the measurements already answer is an engine setting.
- The one recorded exception is `search --method` and `--no-rerank`, which exist to reproduce a row of `MEASUREMENTS.md`.
- `status` and an agent's `status` are the same answer.
- Two readers are bounded and share `tool_views`, so they cannot disagree.
- An agent's client entry names a project and never a directory, so one entry serves that project on every machine where it was initialised.
- `mcp` refuses `--project-root` and `--project`.
- The bridge resolves the project name through `registry.py`.
- A path never reaches a client entry, a generated template, or a bridge invocation.
- A project a machine has not initialised is answered rather than refused.
- The session opens, `status` reports the condition and the `init` command through `blocked_by`, and the other seven tools are absent because there is no corpus behind them.
- `surfaces/mcp.py` declares that second `status`, and `tests/test_bridge.py` holds the property.
- Generation of an answer is out of scope.
- Adding generation would be a deliberate research feature with its own API, citation contract, and tests rather than a patch.
- No bundle operations exist: `export_bundle` and `import_bundle` are not available.

### The state a reader must be able to trust

- One app process serves each configured project root, and every surface shares that process, its project lock, and its single UltraRAG gateway.
- The app is up or it is down.
- A command reaches the running app over the control API.
- A project with no app up is answered in process, because a read of local state must not refuse itself because no daemon is running.
- Every project-owned artifact lives beneath `<project>/.research-rag`, apart from the machine-local launcher symlink.
- `STORAGE.md` carries the layout, the runtime-root rules, and the marker.
- Every path this app names inside a project carries `research-rag`, including the launcher's pid, port, log, and lock files.
- Two products sharing a pid file would let each stop the other's process.
- Only immutable model binaries are shared through the configured user cache.
- Documents, metadata, chunks, vectors, indexes, logs, and query state never sit in global storage.
- Deterministic source IDs, document IDs, chunk IDs, source paths, and locators are preserved.
- A `source_id` follows the normalized source-relative path and survives a byte change.
- A `document_id` identifies a path and a content version.
- Chunk IDs may change when content or chunking changes.
- No document or chunk ID is implied to be permanent across incompatible generations.
- `current.json` switches only when a generation is completely indexed, and an artifact is validated before the pointer names it.
- BM25 and dense stay in the same generation, and a generation is never selected unless both validate.
- Review state stays hand-editable.
- A hand edit is honoured.
- An unknown field, a wrong type, or a non-normalized path is refused with a message naming the problem and never ignored.
- Reviewed metadata is a read-time overlay on the selected generation.
- Reviewed metadata is never written into a generation file.
- A generation is never marked stale because the overlay moved.
- Exclusions are explicit, reversible, project-local, and enforced by every retrieval surface at once.
- No automatic duplicate guessing is added.
- A passage exclusion is a read-time decision like a reviewed metadata change, and it never turns `stale`.
- An `ingest` cannot bring a removed chunk back, so `status` reports how many entries another generation cannot match.
- `requires` never names `ingest` for one such entry.
- `chunk-exclusions.json` is the one file this app writes that the frozen `research-ultra-rag-mcp` does not read, because that product resolves its four files by name.
- While the frozen product is installed, it keeps serving a passage excluded here, and that divergence is accepted rather than accidental.
- A whole-source exclusion is the decision both products honour, and a file the frozen product never looks up is not one it can be broken by.
- The on-disk contract is frozen while `research-ultra-rag-mcp` is installed, because a change here that it cannot parse is a reader answering with state it could not have written.
- `project.json` is the only file the frozen product reads a known subset of.
- `registry.py` is a pointer file: an id, a name, and a root per project.
- `registry.py` never grows into a cache of a project's state, and one project's record never decides what another may read.
- A project's name is the address an agent's entry carries, so a record's name is an interface.
- A record's name is matched exactly, an ambiguous one is refused, and a name is never resolved through an id.
- The layer machinery lives in `config-ultra-rag-mcp`, not in this repository.
- This repository owns `SETTINGS`, `EffectiveSettings`, the packaged `default.toml`, and the three names it resolves its own layers by.
- A settings write reaches `<project>/.research-rag/config.toml` and nothing else.
- A value a layer above the project file supplied is not writable, because a lower file cannot override it.
- The cost of a change is computed from what a generation records rather than from `Setting.layer`, which mislabels keys no generation reads.
- A write merges into the document and never overwrites it.
- Schema and policy versions, the retrieval-method set, and the boundary names stay in code, so a settings file cannot forge what a generation is or where the app may write.
- A tunable that decides what a generation contains enters the retrieval-policy fingerprint or the recorded chunk settings.
- A tunable that cannot decide that is a runtime setting.
- `use_generation` and `remove_generation` match a caller-supplied name against the pattern the builder writes before the name becomes a path, and both take the project lock.
- Both refuse the generation search reads, because nothing refuses a delete loudly.
- An unlinked memory mapping keeps reading the old bytes, and a later open returns a missing-file error.
- `remove_generation` also refuses a generation a pending activation has named, and it requires the id twice.
- The durable-write order is required and expected.
- Directory fsyncs are grouped inside a unit, and a commit point never becomes durable before the artifacts it describes.
- A file handed off to a peer process and rewritten before every use may skip fsync.
- A checkpoint is durable per bounded batch rather than per phase.
- A redone unit is one batch, never a phase.
- `ingest` is one build per project at a time.
- An operation arriving while another process holds the project lock is rejected promptly and names the resident build's phase.
- An operation is never queued for minutes, because the workspace is waiting on it.
- Extraction and vectors are reused only for a source with matching bytes, storage policy, and processing fingerprints.
- A vector is reused only when the canonical text, model revision, and dimension match exactly.
- `force_recompute=True` disables all reuse.
- The dense backend is recorded in the manifest and dispatched from that record.
- A generation is never rebuilt with a different backend.
- A selected PDF/EPUB that fails extraction is never silently skipped.
- A selected source that fails extraction fails the new generation and leaves the current one intact.

### Boundaries a contributor must not cross

- The package stays split at `surfaces/`.
- The engine imports neither Starlette nor FastMCP, no engine module imports a surface, and no surface imports another.
- `tests/test_architecture.py` enforces that split.
- Neither surface re-declares the other's tool, resource, or operation, because a second copy of an operation is a second place for it to be wrong.
- The workspace's profile turns off what it does not serve, and `ResearchUIAdapter._arguments` drops every argument an operation does not accept.
- A workspace control never travels as an argument the app ignores.
- The shared UI is pinned by commit.
- The adapter, source authorization, and the process host stay in this package, and the shared static workspace is never copied into it.
- Workspace code never reads or mutates generation artifacts directly.
- Workspace code calls `ResearchService`, except for safely serving an allowlisted original from the source root.
- Browser and control write endpoints stay same-origin, JSON-only, and loopback-only.
- An arbitrary filesystem output path is never accepted.
- Ingestion selects only regular PDF and EPUB files beneath the configured sources directory.
- Source symlinks and path traversal are rejected.
- The underlying vanilla operations are never exposed, UltraRAG is never patched, and the dense backends never move into the gateway.
- Every answer goes through the core.
- The workspace and the agent surface call a `ResearchService` method.
- The command line reaches that same service over the control API and builds one in process only when the project has no app up.
- One capability has one implementation, and the three surfaces cannot disagree.
- Settings are the workspace's to change and the command line's to read, never an agent's.
- `research-rag config` stays read-only.
- A control settings write carries the same-origin, JSON-only, and loopback-only gates a browser write does.
- The workspace stays bound to loopback.
- Remote exposure and authentication assumptions never appear without an explicit security design.
- A child process never receives `dict(os.environ)`; `child_process_environment()` is used instead.
- This app is never pointed at a second copy of itself.
- A process is signalled only when this project can prove it owns it.
- The stop sweep requires this app's entry point in the command line *and* this project as `--project-root`.
- The stop sweep never signals the process doing the sweep.
- The author's own projects, notes, drafts, and every original PDF/EPUB stay out of git.
- A new personal location is added to the block in `.gitignore`.
- An unimplemented feature keeps its explicit limitation, and planned work is never presented as shipped.

### Reporting a condition

- Every condition a reader must act on is named, and its remedy sits beside it.
- `health.py` owns one check per condition in one of four states: `ok`, `warn` (the answer is worse), `blocked` (the app cannot work), and `unknown` (the check did not run, which never reads as healthy).
- Every check reads: no process, no network, and no write, so the report is built before the gateway is opened.
- A condition the report cannot state reads as unknown rather than being omitted.
- `blocked_by` and `degraded` are omitted when empty and both present together when a project is healthy.
- A reader who sees one of them can name the fix without reading this file.
- A reason is a sentence, and a remedy is a pasteable command.
- The per-check detail is a `doctor` reader.
- `doctor` reads the same report, so the two surfaces cannot disagree.
- `doctor` writes nothing at all.
- `--prefetch-models` and `--repair-runtime` are the only operations that reach the network.
- A repair that discards evidence moves it aside.
- A failure message names the reason and where to read the rest, so `Connection closed` is never the whole of what a reader is told.
- No process-tree inspection is done, because the vanilla tool name says which component was busy and that component writes its own log.
- A gateway that cannot start is reported by the operation that needed it, never by a surface that never appeared.
- The app resolves its settings and starts serving before it imports the retrieval stack or opens the gateway, so a project reads fine where the runtime is not installed.
- `tests/test_integration.py` pins that order.
- Blocking extraction and filesystem scans stay outside the event loop.
- Every project operation is serialized with both the in-process lock and the cross-process `project.lock`.

## Architecture

```text
MCP client ── HTTP /mcp ─┐
                       │
stdio client ── bridge ──┤  --project-name, never a path
                       ├── research-rag (one process, one port, one lock)
Local browser ── HTTP ──┤      │
                       │      ├── ResearchService
Terminal ── control ───┘      ├── shared workspace (surfaces/ui.py)
                              ├── agent tools and resources (surfaces/mcp.py)
                              ├── control API (control.py)
                              ├── client registry (app.py)
                              │
                              ├── project record (registry.py)
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

- The vanilla gateway is an implementation dependency below the app, not a second surface.
- The shared workspace package owns no research state.
- The stdio bridge is a transport, not a second copy of the tools.

## Working rule

- Use `pathlib.Path`, type hints, and JSON-serializable payloads.
- A file under `evaluation/` changes only with the protocol that file describes.
- Every change is committed and pushed without waiting to be asked.
- The validation below runs first.
- The commit message says what the change does and why.
- The change is pushed to `origin main` before anything else starts.
- A change that is not on the remote is lost, and the collection cannot record a pointer to a commit that is not there.

## Validation

```bash
uv lock --check
uv run ruff format --check .
uv run ruff check .
uv run pytest -q
uv run python -m compileall -q src tests
```

- A retrieval-quality claim is produced by `uv run python scripts/evaluate_retrieval.py --project <project> --offline`, with `--validate-only` added first.
- The result of that harness is recorded in `MEASUREMENTS.md`.
- A documented retrieval default never changes on an unrecorded run.
- A workspace adapter change covers safe source-file resolution, the arguments each operation forwards, and the real host against an existing project without mutating its sources.
- Shared workspace, JSON validation, capability, and same-origin changes belong in `ui-ultra-rag-mcp` and must pass that package's own tests before the pinned commit here is updated.
- A source or retrieval change keeps the integration test launching the real vanilla gateway, building both indexes, running hybrid and dense search, retrieving the known passage, and proving a neighbouring file was excluded, before it restarts offline and repeats hybrid reranked search from the caches.
- A source or retrieval change keeps unit tests covering RRF and failure atomicity without model downloads.
- Those unit tests also cover no-op ingestion, additions, changes, removals, reviewed metadata and exclusions, byte changes that preserve size and mtime, forced regeneration, vector reuse, and final artifacts.
- Those unit tests also prove that post-ingestion metadata corrections immediately affect every read surface and filter mode without modifying generation files.
- A significant extraction change also tests a representative real collection without writing into its source directory.