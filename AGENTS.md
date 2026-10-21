# AGENTS.md

## Scope

- Project-scoped PDF/EPUB/MOBI evidence retrieval, and white-box syntheses rendered only from that evidence.
- One process and port per project, shared by the workspace, MCP, and CLI. Loopback is the default; explicit home-LAN mode exposes only the workspace.
- Ship the project MCP surfaces, workspace, and settings layer here. Do not depend on a sibling project.

## Documentation

| File | Owns |
|---|---|
| `README.md` | The quickstart: requirements, installation, a first project, a first search, the three front ends, and the limits a first reader hits. |
| `surfaces/cli/help.py` | The command reference: `HELP_GROUPS` for every command, `HELP_TOPICS` for a subject, and each command's own usage. |
| `STORAGE.md` | Files, fields, portability, and hand edits. |
| `FEATURES.md` | Shipped and excluded capabilities, and UltraRAG reuse. |
| `MEASUREMENTS.md` | Measurement protocols, the operating envelope, and tunable mechanisms. |
| `evaluation/README.md` | The judged query set and the harness. |
| `ROADMAP.md` | Deferred ideas. |
| `TODO.md` | Open work. |
| `AGENTS.md` | Engineering constraints. |
| Folder `README.md` files | Each source and test folder's boundaries. |
| Module docstrings | Local mechanisms and boundaries. |

- Give each fact one owner.
- Read constants, models, revisions, and thresholds from `default.toml` and code, not from markdown.
- The README's examples parse against the CLI parser, and `tests/gates/test_documentation.py` holds that.
- A command added to the parser must reach `HELP_GROUPS` in the same change. The same gate holds that.
- Markdown states current rules, not review logs, completed tasks, or superseded decisions. Git holds history.
  - Remove finished TODOs and superseded measurements.
- Re-measure changes to corpus, extraction, or retrieval defaults, and update every affected figure.
- Reference file paths or harness names, not section numbers.
- Keep the child README standalone, without sibling comparisons or links.
- Write active verbs and concrete nouns, one fact per sentence.
  - Cut preambles, hedges, sales language, repeated headings, and filler such as *very*, *really*, *actually*, *simply*, *just*, *importantly*, and *of course*.
  - Use bullets for lists. Delete sentences that add no rule or fact.
- Gates: `tests/gates/test_documentation.py` and `tests/core/test_review_state_edits.py`.

## Presenting decisions to the user

- Present numbered, concrete options with identifiers, smallest change, cost, risk, research-contract effects, and reversibility.
- Recommend one option with a reason. Include "no change" when work can proceed without a choice.
- Get a choice before changing generation artifacts, retrievable evidence, portable state, or on-disk layouts.

## Release and install

- Classify a release by compatibility: major for an incompatible supported contract; minor for compatible behavior changes; patch for a small compatible fix without behavior changes.
- `pyproject.toml` owns the version and Python range. The runtime version reads installed distribution metadata.
- `update` compares published releases, not branch heads.
  - It shows the matching GitHub changelog before approval.
  - Declining, or an unavailable approval, must not install, stop apps, or change project state.
  - `tests/runtime/test_release.py` validates release versions and local tag targets.
- `--version` and `update` share app, installed, and restart reporting.
- `install` writes PATH and desktop entries, not a serving process.
- `install`, `update`, `help`, `api-key`, and `--version` need no project.
- `LICENSE` grants Apache-2.0 for this code. `NOTICE` owns separate dependency terms, including AGPL-3.0 extraction.
- Preserve `USER_CONFIG_DIRECTORY = research-rag`, `SETTINGS_ENVIRONMENT_PREFIX = RESEARCH_RAG_`, and the `research-rag` model cache.
  - Do not use these names for another product. `tests/project/test_data_roots.py` covers them.
- Dependency pins live in `pyproject.toml`. Embedding and reranker registries own model revisions.
  - `corpus/docling_env.py` owns the separately installed Docling runtime pins.
  - Keep the `bm25s` non-ASCII stopword fork until upstream fixes round-tripping. Reject invalid lists promptly.
  - The Docling worker's page-label probe falls back to the physical page on `Page.get_label()`'s `IndexError`. Narrow the guard when upstream issue 5140 closes.
- `synthesis/render.py`, `ops.py`, `pieces.py`, and `checks.py` copy the white-box-synthesis renderer from `AhmedKishki/agent-skills`. The copy is deliberate; carry a fix to both.

## Rules

### The contract a reader relies on

- Preserve prominent UltraRAG credit, upstream links, licences, `NOTICE`, and the independent-project disclaimer.
  - Credit THUNLP, NEUIR, OpenBMB, AI9stars, and contributors without implying endorsement.
- No original source document is edited or written. `sources/` is the authority for a quotation.
- `text` in a result is cleaned semantic text, never a transcript.
- A result carries no per-passage quotation flag. The cleaned-text rule is stated once in the contract, not repeated on every passage.
- An agent gets nine tools and one resource with bounded answers, never a corpus inventory.
  - The workspace and CLI `sources` own inventory access.
- A generation the app cannot serve is stated as `hybrid_ready: false` beside `generation_upgrade_required`. The old generation stays searchable with a warning.
  - Report an upgrade requirement only for a serving incompatibility, not a newer extraction, cleaning, or ranking policy. Preserve compatible generations; expose policy differences through verbose diagnostics and DEBUG logs.
  - A source snapshot change or read-time exclusion does not require discarding the selected generation. Distinguish optional ingestion from required recovery.
- Expose only caller decisions, not engine tuning.
  - Agents get no retrieval mode, output view, or chunk tuning.
  - CLI `search --method` and `--no-rerank` are diagnostic exceptions.
- CLI and agent `status` share bounded `tool_views` projections.
  - Allowlist actionable app fields. Never merge process or client inventories into MCP answers.
- Client entries and generated templates carry a project name, never a path.
  - `mcp --project-name` or `RESEARCH_RAG_PROJECT_NAME` resolves exactly through `registry.py`.
  - Refuse `mcp --project-root` and `mcp --project`.
- The bridge decides nothing at launch. It asks per call whether the app answers, serves the same eight tools while it does not, and never starts it.
- Writes take turns one at a time, and searches `runtime.search_concurrency` at a time. Waiting callers are served in rounds by agent, waits are bounded, and a refusal says how many were ahead.
  - A second `ingest` during a build is refused with the build's phase, not queued.
- For an uninitialized project, open the session with only `status`, an `init` remedy in `blocked_by`, and no corpus tools.
  - `tests/surfaces/test_bridge.py` covers this surface in `surfaces/mcp.py`.
- `synthesise` is the one operation that writes an answer. It is white-box only.
  - DeepSeek's API is its only provider. A set key enables it in every project on the account. `runtime.offline` refuses it.
  - The model reads the corpus through `search` and `get_passage` and submits a derivation. It never writes answer prose.
  - The app builds a derivation's sources from the passages the run retrieved. The model names a source by its label only.
  - `synthesis/render.py` renders the derivation and rejects wording no source supplies. A rejected derivation returns to the model up to `synthesis.max_attempts` times and is never shown.
  - A missing link is a `⟦GAP⟧`. No AI-authored link enters a synthesis.
  - When no source supplies a step, the result is `generated: false` and carries no model text.
  - The question, the system prompt, and every passage the run retrieves leave the machine. Each result names the endpoint and how many passages it sent; a verbose result names their `chunk_id`s.
  - A synthesis reasons at `synthesis.reasoning` unless the caller chooses `none`, `low`, `high`, or `max`.
  - Syntheses are not saved.
  - An agent's `synthesise` returns the synthesis, the references, and what was sent; `verbose` adds the rest. The command line synthesises the same way, and the workspace shows everything with provenance and reasoning collapsed.
  - The agent instructions say to call `synthesise` only when the user asks for a synthesis.
- No bundle operations exist. `export_bundle` and `import_bundle` are not available.

### The state a reader must be able to trust

- One app process serves each configured project root. Every surface shares that process, its project lock, and its single direct retrieval backend.
  - Implement capabilities once in `ResearchService`. Running CLI commands use the control API.
  - Without an app, answer local-state operations in process.
- Keep project artifacts under `<project>/.research-rag`, including namespaced pid, port, and lock files.
  - `STORAGE.md` owns runtime relocation and its marker.
  - Share only immutable model binaries and versioned dependency runtimes through the user cache, never corpus, index, log, or query state.
- Preserve deterministic IDs, normalized source-relative paths, and locators.
  - `source_id` survives byte changes. `document_id` identifies path plus content version.
  - Chunk IDs can change with text or chunking. Document and chunk IDs need not survive incompatible generations.
- Activate `current.json` only after complete indexing and validation of both BM25 and dense artifacts in the same generation.
- Honor hand-edited review state. Reject unknown fields, wrong types, and non-normalized paths explicitly.
- Apply reviewed metadata and source and passage exclusions at read time, without rewriting artifacts or marking the generation `stale`.
  - An excluded source is never ingested. Data a generation already holds for it is carried into later generations, filtered from every search.
  - Only `uningest` takes a source's chunks and data out of a generation. Its file, reviewed metadata, and exclusion stay.
- Keep source and passage exclusions explicit, reversible, project-local, and consistent across surfaces.
  - Do not guess duplicates automatically.
  - Report unmatched passage exclusions in `status`. `requires` must not prescribe `ingest` to recover a removed chunk.
- This app alone owns portable `project.json`, `source-metadata.json`, `source-exclusions.json`, and `source-catalog.json`.
  - Keep passage decisions separate in `chunk-exclusions.json`.
  - Write no machine-specific portable state except the relocated-runtime record.
- `registry.py` holds project id, name, and root, never cached project state or cross-project permissions.
  - Resolve names exactly. Refuse ambiguity. Never resolve a name through an id.
- Own `settings_layers`, `SETTINGS`, `EffectiveSettings`, `default.toml`, and layer-location names here.
  - Settings writes merge only into `<project>/.research-rag/config.toml`.
  - Refuse writes to keys overridden by higher layers.
  - Determine rebuild cost from generation records, not `Setting.layer`.
- The DeepSeek key is a credential, not a setting.
  - It lives in the account's `credentials.json`, or in `RESEARCH_RAG_DEEPSEEK_API_KEY`, which wins. `STORAGE.md` owns the file.
  - It never enters a project, a `config.toml`, a payload, a log, or an error.
  - Only this machine's command line and workspace set or clear it. `/api/credentials` is local-only.
- Keep schema and policy versions, retrieval-method sets, and boundary names in code.
  - Artifact-affecting tunables enter retrieval fingerprints or recorded chunk settings. Other tunables are runtime settings.
- Validate generation names against the builder's pattern before forming paths. Lock `use_generation` and `remove_generation`.
  - Refuse operations on the generation being searched, and removal of one pending activation.
  - Require the removal id twice. Preserve open-reader safety.
- Make artifacts durable before commit points, with directory fsyncs grouped per unit.
  - Only disposable peer handoffs rewritten before every use may skip fsync.
  - Checkpoint bounded batches, not whole phases.
- Allow one ingestion build per project. Reject contention promptly and name the resident phase instead of queuing.
- Reuse extraction only when bytes, storage policy, and processing fingerprints match.
  - Ordinary ingestion may carry an unchanged PDF's complete existing artifacts forward with their recorded processing provenance. This preserves evidence; it does not reuse custom extraction under a Docling fingerprint. `force_recompute` disables this passthrough.
  - Record each PDF's actual extractor when a build mixes retained legacy artifacts with new Docling output. Never relabel a selected generation or claim its target backend produced every retained document.
  - Decide extraction/chunk reuse and vector reuse independently. A processing change must not discard a valid exact-input vector, and a model change must not discard compatible extracted text.
  - Vector reuse also requires exact canonical text, model revision, and dimension.
  - `force_recompute=True` disables all reuse for the sources a build reads. A source a build carries is never read.
- `ingest` reads every source that is not excluded, or only the sources the command line names; a build that names sources carries every other held source as it is, and records the bytes it was built from.
  - Naming sources and `uningest` are command-line only. The agent's `ingest` builds every source.
- Dispatch dense backends from the manifest. Never substitute one during reuse.
- A source-local PDF/EPUB/MOBI extraction failure omits that source from a validated partial generation when at least one source remains. Retain the partial generation without selecting it; disclose the omission, mark a selected partial generation stale, and retry omitted sources on ordinary ingestion. Other failures preserve the selected generation.

### Boundaries a contributor must not cross

- Keep the engine free of Starlette, FastMCP, and surface imports. Surfaces must not import or redeclare one another.
  - `tests/gates/test_architecture.py` enforces the split.
- Ship the workspace in `surfaces/workspace/` with its adapter, source authorization, and host.
  - Confine each browser workspace to its serving project and address. Never list, open, start, or switch to another project from the browser. The command line owns account-wide project selection.
  - A LAN viewer is not shown the panels that act on this machine. `ResearchUIAdapter._arguments` removes unsupported arguments.
  - A LAN viewer may run `synthesise`. It never sees or sets the key.
  - Call `ResearchService`. Never read generation artifacts directly. Allowlisted originals are the sole file-serving exception.
- Browser and control writes must be same-origin and JSON-only, with no caller-selected output paths. Control and MCP remain loopback-only.
  - Explicit home-LAN mode accepts private IPv4 peers and detected local-interface hosts. Remote browser writes require an exact validated HTTP origin; cross-site writes are refused.
  - Home-LAN HTTP has no authentication or encryption. Other remote exposure requires an explicit security design.
- Ingest only regular PDF/EPUB/MOBI files beneath the configured source root. Reject symlinks and traversal.
- Do not import or install UltraRAG, acquire its source runtime, or expose upstream stage operations. Preserve its attribution and historical artifact identifiers without a runtime dependency.
  - Retired gateway/runtime console names provide side-effect-free migration guidance. They never start a server, download, validate, delete, or quarantine an upstream cache.
- Preserve CLI access to every workspace capability.
  - Nothing is configurable in the workspace that the command line and `config.toml` cannot also configure. The DeepSeek key is the exception: the command line sets it, and no `config.toml` holds it.
  - Settings writes belong to the workspace and the command line, never agents. `research-rag config --write` takes the workspace's path through the service.
  - A running app takes a change from any of the three at its next call. An edit that does not resolve keeps the values in force and is named in the health report.
  - Open a browser only with `--start-ui`.
- Use `child_process_environment()`, never `dict(os.environ)`, for subprocesses.
- Never point the app at a second copy of itself.
- Signal only proven project-owned processes. Require the app entrypoint and a matching `--project-root` in stop sweeps.
  - A recorded app is proven instead by the app entrypoint and the listening socket on its recorded port, so an app started by `--project` or a bare call can be stopped.
  - Never signal the sweep itself.
- Serve from the starting terminal. Never adopt a serving process another terminal, or none, holds; restart it here when asked.
  - Refuse to serve from a process with no controlling terminal.
  - Closing the starting terminal ends its app, by SIGHUP or by the lost terminal. Only `stop` or a takeover ends a detached app.
  - Read terminal ownership from `/proc/<pid>/stat`, not only `research-rag-ui.tty`.
  - `projects` and `stop` report the condition. They never launch a second app.
  - A bare call or `start` asks an interactive terminal whether to take the app over, and `--take-over` takes it without asking. Otherwise it reports the condition with both remedies.
  - A takeover stops the app through `stop`'s ownership proof, waits for it to exit, and serves on its port. It refuses while a write holds the project and never launches a second app.
  - `tests/runtime/test_attached_workspace.py` covers this. Detached serving is deferred in `TODO.md`.
- Keep private projects, notes, drafts, and original PDFs/EPUBs/MOBIs out of Git. Add new personal paths to `.gitignore`.
- State unimplemented limits explicitly, never as shipped capabilities.

### Reporting a condition

- Name actionable conditions with a reason, a pasteable remedy, and the relevant log path. Never report only `Connection closed`.
- `health.py` owns read-only checks: `ok`, `warn`, `blocked`, or `unknown` when unchecked.
  - Start no process, network request, or write while building health reports.
  - Omit empty `blocked_by` and `degraded`.
- Default `doctor` reads the same report without writing.
  - Only explicit `--prefetch-models` may provision models or the Docling environment. `--repair-runtime` remains an informational no-op, including offline.
- Diagnose component failures through the requesting operation and its logs, not process-tree inspection.
- Resolve settings and serve before loading retrieval dependencies. Local reads must work without a gateway executable.
  - Lightweight commands must not require the retrieval stack.
- Move blocking extraction and filesystem scans off the event loop.
- Serialize project writes with both the in-process lock and the cross-process `project.lock`. Bound both waits and refuse after them.
  - Reads take neither lock. A read holds a lease on the generation it resolved, and removing that generation waits for it.
  - The direct backend holds one BM25 retriever. A build and a search take it in turn.
- Search counts store ranks, ids, times, result counts, and who asked. They store the question and its filters only while `runtime.search_history` is on, and `history --clear` removes them and leaves the counts. A measurement does not count.
- Extraction reads existing PDF, EPUB, or MOBI text only. Scanned PDFs need OCR performed outside this app; no OCR command, agent tool, control route, or workspace action exists.
- MOBI parsing runs offline in a resource-bounded child without conversion or resource output. Reject DRM and unsupported or corrupt containers; retain logical-part locators rather than inventing printed pages. `FEATURES.md` owns supported variants and safety limits.
- Docling is the only PDF extractor. PDF structural cleanup uses its labels, content layers, and reading order before chunking; the custom reader and native fallback are retired.
  - Existing generations remain readable. The retirement alone never starts ingestion or reprocesses unchanged PDFs. New or changed PDFs use Docling; full reprocessing requires `force_recompute`.
  - Preserve declared captions, formulas, tables, code, explanatory footnotes, and unknown content. A reference label still needs bibliographic structure before removal, and only a declared heading opens an apparatus section.
  - Local normalization and corruption/loss checks consume Docling text. Irrecoverable lexical tokens leave visible gaps and loss diagnostics; no alternate PDF extractor repairs them.
  - Recognised formatting and math glyphs are not corruption. Harmless symbol-only fragments may be omitted as non-evidence and must be counted separately.
  - `ingestion.maximum_unclean_percent` caps cumulative substantive-character loss per PDF document after confirmed furniture and non-evidence are removed.
  - Over-budget or completely unreadable PDFs are omitted through the source-local partial-generation path. They never alter the selected generation automatically.
  - Preserve EPUB's existing unit and source quality rules.
  - Keep physical PDF page locators. Do not infer printed numbers from recovered text.
  - Docling runs in a resource-bounded child with OCR, VLM, and every enrichment off. It is never installed into this app's environment: an online first PDF conversion builds a pinned, versioned managed environment under the user cache, in isolation, and fetches models lazily; an offline conversion never fetches and needs a ready environment or fails closed. `doctor --prefetch-models` provisions environment and models ahead of an offline run, and a ready environment or cached model is reused without a fetch. Serving or carrying forward existing evidence never provisions Docling.
- Opening a source asks the desktop's own viewer through an authorised path, and falls back to the browser only where there is no desktop.

## Architecture

```text
MCP client ── HTTP /mcp ─┐
                        │
stdio client ── bridge ──┤  --project-name, never a path
                        ├── research-rag (one process, one port, one lock)
Local browser ── HTTP ──┤      │
                        │      ├── ResearchService
Terminal ── control ───┘      ├── the workspace and its adapter (surfaces/ui.py,
                               │   surfaces/workspace/)
                               ├── agent tools and resources (surfaces/mcp.py)
                               ├── control API (runtime/control.py)
                               ├── client registry (runtime/app.py)
                               ├── synthesis (synthesis/), the one call that leaves
                               │   the machine: HTTPS to DeepSeek
                               │
                               └── the engine, using direct compatible chunk/BM25
                                   retrieval (retrieval/direct.py)
```

### Layer ownership

- Each source folder owns one concern. Its boundaries are in its README, and its tests are in the matching test folder.
- Folders are layered, lowest first: `storage`, `project`, `corpus`, `retrieval`, `generations`, `synthesis`, `core`, `runtime`, `surfaces`. A module imports from its own folder and the folders below it.
  - `runtime/app.py` composes the surfaces it serves, and `runtime/control.py` reads the workspace's write guard. These are the only upward imports.
  - `tests/gates/test_architecture.py` enforces the order.
- Update test paths when renaming folders.
- Import shared concerns instead of copying them, including state names in `project/state_files.py` and shared surface contracts.
- Error types, schema versions, and boundary names must be importable without fastembed.

## Working rules

- Use `pathlib.Path`, type hints, and JSON-serializable payloads.
- A file under `evaluation/` changes only with the protocol it describes.
- Validate before committing, then push each change to `origin main` before starting the next.
  - Commit messages state what changed and why.
- End work with a clean tree: `git status --short` reports nothing.
  - Work that arrived in the tree from elsewhere counts as part of it. Commit it with this work, or state why not.

## Validation

- Keep one primary test for each supported behavior or failure mechanism.
  - Extend an existing behavioral test when it already exercises the same input and assertion.
  - Add parameter cases for distinct boundaries, not equivalent spellings or arbitrary sample values.
  - Do not freeze cosmetic wording, obsolete filenames, or private implementation spellings when they do not express a supported contract.
- During ordinary edits, run `uv run pytest -q -m "not integration"`.
- `integration` covers external runtimes, process contention, live transport waits, and shipped scripts.
  - Cheap interpreter checks and shortened in-process waits stay in the fast subset.
- Run `uv run pytest -q -m integration` when changing an integration boundary.
- The full suite remains required before committing. CI runs both subsets.

```bash
uv lock --check
uv run ruff format --check .
uv run ruff check .
uv run pytest -q
uv run python -m compileall -q src tests
```

- Produce a retrieval-quality claim with `uv run python scripts/evaluate_retrieval.py --project <project> --offline`, with `--validate-only` added first.
  - Record valid measurements in `MEASUREMENTS.md`. Never change documented defaults on an unrecorded run.
- Workspace changes test safe source resolution, argument forwarding, and the real host, without source mutations. Run the `surfaces/workspace/` tests.
- Source and retrieval integration tests use the real direct backend, both indexes, hybrid and dense search, a known passage, and an excluded neighbour.
  - Restart offline and repeat a hybrid reranked search from caches.
- Offline unit tests cover:
  - RRF, failure atomicity, and no-op ingestion.
  - Source additions, changes, and removals, including byte changes that preserve size and mtime.
  - Forced regeneration, reuse, and final artifacts.
  - Reviewed metadata and exclusions across every read surface and filter, without rewriting generation files.
- Significant extraction changes also test a representative real collection without writing sources.
