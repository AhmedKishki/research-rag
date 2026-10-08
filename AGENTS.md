# AGENTS.md

## Scope

- Project-scoped PDF/EPUB evidence retrieval, not answer generation.
- One process and loopback port per project, shared by the workspace, MCP, and CLI.
- Ship the gateway, workspace, and settings layer here. Do not depend on a sibling project.

## Documentation

| File | Owns |
|---|---|
| `README.md` | The quickstart: requirements, installation, a first project, a first search, the three front ends, and the limits a first reader hits. |
| `help` in `surfaces/cli.py` | The command reference: `HELP_GROUPS` for every command, `HELP_TOPICS` for a subject, and each command's own usage. |
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

- `pyproject.toml` owns the version and Python range. The runtime version reads installed distribution metadata.
- `update` compares published releases, not branch heads.
  - It shows the matching GitHub changelog before approval.
  - Declining, or an unavailable approval, must not install, stop apps, or change project state.
  - `tests/runtime/test_release.py` validates release versions and local tag targets.
- `--version` and `update` share app, installed, and restart reporting.
- `install` writes PATH and desktop entries, not a serving process.
- `install`, `update`, `help`, and `--version` need no project.
- `LICENSE` grants Apache-2.0 for this code. `NOTICE` owns separate dependency terms, including AGPL-3.0 extraction.
- Preserve `USER_CONFIG_DIRECTORY = research-rag`, `SETTINGS_ENVIRONMENT_PREFIX = RESEARCH_RAG_`, and the `research-rag` model cache.
  - Do not use these names for another product. `tests/project/test_data_roots.py` covers them.
- Pins live in `pyproject.toml` and `gateway/manifest.py`.
  - Keep the `bm25s` non-ASCII stopword fork until upstream fixes round-tripping. Reject invalid lists promptly.
  - `_pdf_locator` falls back to the physical page on `Page.get_label()`'s `IndexError`. Narrow the guard when upstream issue 5140 closes.

## Rules

### The contract a reader relies on

- Preserve prominent UltraRAG credit, upstream links, licences, `NOTICE`, and the independent-project disclaimer.
  - Credit THUNLP, NEUIR, OpenBMB, AI9stars, and contributors without implying endorsement.
- No original source document is edited or written. `sources/` is the authority for a quotation.
- `text` in a result is cleaned semantic text, never a transcript.
- A result carries no per-passage quotation flag. The cleaned-text rule is stated once in the contract, not repeated on every passage.
- An agent gets eight tools and one resource with bounded answers, never a corpus inventory.
  - The workspace and CLI `sources` own inventory access.
- A generation the app cannot serve is stated as `hybrid_ready: false` beside `generation_upgrade_required`. The old generation stays searchable with a warning.
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
- Answer generation is out of scope until the design in `ROADMAP.md` is chosen.
- No bundle operations exist. `export_bundle` and `import_bundle` are not available.

### The state a reader must be able to trust

- One app process serves each configured project root. Every surface shares that process, its project lock, and its single UltraRAG gateway.
  - Implement capabilities once in `ResearchService`. Running CLI commands use the control API.
  - Without an app, answer local-state operations in process.
- Keep project artifacts under `<project>/.research-rag`, including namespaced pid, port, and lock files.
  - `STORAGE.md` owns runtime relocation and its marker.
  - Share only immutable model binaries through the user cache, never corpus, index, log, or query state.
- Preserve deterministic IDs, normalized source-relative paths, and locators.
  - `source_id` survives byte changes. `document_id` identifies path plus content version.
  - Chunk IDs can change with text or chunking. Document and chunk IDs need not survive incompatible generations.
- Activate `current.json` only after complete indexing and validation of both BM25 and dense artifacts in the same generation.
- Honor hand-edited review state. Reject unknown fields, wrong types, and non-normalized paths explicitly.
- Apply reviewed metadata and passage exclusions at read time, without rewriting artifacts or marking them `stale`.
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
  - Vector reuse also requires exact canonical text, model revision, and dimension.
  - `force_recompute=True` disables all reuse.
- Dispatch dense backends from the manifest. Never substitute one during reuse.
- A source-local PDF/EPUB extraction failure omits that source from a validated partial generation when at least one source remains. Retain the partial generation without selecting it; disclose the omission, mark a selected partial generation stale, and retry omitted sources on ordinary ingestion. Other failures preserve the selected generation.

### Boundaries a contributor must not cross

- Keep the engine free of Starlette, FastMCP, and surface imports. Surfaces must not import or redeclare one another.
  - `tests/gates/test_architecture.py` enforces the split.
- Ship the workspace in `surfaces/workspace/` with its adapter, source authorization, and host.
  - Hide unsupported capabilities. `ResearchUIAdapter._arguments` removes unsupported arguments.
  - Call `ResearchService`. Never read generation artifacts directly. Allowlisted originals are the sole file-serving exception.
- Browser and control writes must be same-origin, JSON-only, and loopback-only, with no caller-selected output paths.
  - Remote exposure requires an explicit security design.
- Ingest only regular PDF/EPUB files beneath the configured source root. Reject symlinks and traversal.
- Do not patch UltraRAG, expose vanilla operations, or move dense backends into the gateway.
- Preserve CLI access to every workspace capability.
  - Settings writes belong to the workspace, not agents. `research-rag config` is read-only.
  - Open a browser only with `--start-ui`.
- Use `child_process_environment()`, never `dict(os.environ)`, for subprocesses.
- Never point the app at a second copy of itself.
- Signal only proven project-owned processes. Require the app entrypoint and a matching `--project-root` in stop sweeps.
  - Never signal the sweep itself.
- Serve from the starting terminal. Report a detached serving process instead of adopting it.
  - Refuse to serve from a process with no controlling terminal.
  - Closing the starting terminal ends its app, by SIGHUP or by the lost terminal. Only `stop` ends a detached app.
  - Read terminal ownership from `/proc/<pid>/stat`, not only `research-rag-ui.tty`.
  - `start`, bare invocation, `projects`, and `stop` report the condition with a `stop` remedy. They never launch a second app.
  - `tests/runtime/test_attached_workspace.py` covers this. Detached serving is deferred in `TODO.md`.
- Keep private projects, notes, drafts, and original PDFs/EPUBs out of Git. Add new personal paths to `.gitignore`.
- State unimplemented limits explicitly, never as shipped capabilities.

### Reporting a condition

- Name actionable conditions with a reason, a pasteable remedy, and the relevant log path. Never report only `Connection closed`.
- `health.py` owns read-only checks: `ok`, `warn`, `blocked`, or `unknown` when unchecked.
  - Start no process, network request, or write while building health reports.
  - Omit empty `blocked_by` and `degraded`.
- Default `doctor` reads the same report without writing.
  - Only explicit `--prefetch-models` and `--repair-runtime` may fetch or write. Quarantine discarded evidence.
- Diagnose gateway failures through the requesting operation and component logs, not process-tree inspection.
- Resolve settings and serve before opening the gateway. Local reads must work without a managed runtime.
  - Lightweight commands must not require the retrieval stack.
- Move blocking extraction and filesystem scans off the event loop.
- Serialize project writes with both the in-process lock and the cross-process `project.lock`. Bound both waits and refuse after them.
  - Reads take neither lock. A read holds a lease on the generation it resolved, and removing that generation waits for it.
  - The gateway holds one BM25 retriever. A build and a search take it in turn.
- Search counts store ranks, ids, times, result counts, and who asked. They store the question and its filters only while `runtime.search_history` is on, and `history --clear` removes them and leaves the counts. A measurement does not count.
- Extraction reads a PDF or EPUB text layer only. Scanned PDFs need OCR performed outside this app; no OCR command, agent tool, control route, or workspace action exists.
- PDF cleaning runs locally. Unhealthy PDF blocks trigger the bundled native text extractor automatically; irrecoverable lexical tokens are omitted with visible gaps and loss diagnostics.
  - `ingestion.maximum_unclean_percent` caps cumulative substantive-character loss per PDF document after confirmed furniture and non-evidence are removed.
  - Over-budget or completely unreadable PDFs are omitted through the source-local partial-generation path. They never alter the selected generation automatically.
  - Preserve EPUB's existing unit and source quality rules.
  - Keep physical PDF page locators. Do not infer printed numbers from recovered text.
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
                               │
                               └── the engine, ending in the stdio MCP gateway
                                   the vanilla gateway proxies (gateway/)
```

### Layer ownership

- Each source folder owns one concern. Its boundaries are in its README, and its tests are in the matching test folder.
- Import within the module's folder or from the folder above, not sideways between sibling folders.
- Update test paths when renaming folders.
- Import shared concerns instead of copying them, including state names in `project/state_files.py` and shared surface contracts.
- Error types, schema versions, and boundary names must be importable without fastembed.

## Working rules

### Active PDF recovery coordination

- This work fixes false-positive PDF corruption detection and safety/correctness issues in automatic native text recovery.
- Ordinary figures, bullets, icons, and formatting symbols must not count as text corruption.
- Damaged text must trigger bounded recovery or local cleaning, not rejection of an otherwise readable PDF.
- Completely unreadable documents remain ineligible.
- Harmless symbol-only fragments may be omitted as non-evidence, but must not be reported as corrupt.
- Current changes cover:
  - `corpus/text_normalization.py`: font-aware recovery of recognised Symbol/dingbat glyphs.
  - `corpus/text_quality.py`: recognised formatting glyphs do not trigger corruption checks; genuinely unknown glyphs and damaged characters still do.
  - `corpus/extraction.py`: span-font glyph recovery and separate benign symbol-only omission and corrupt-passage counters.
  - `generations/ingestion.py`, `core/stats.py`, and `core/tool_views.py`: propagation of separate omission counters.
  - `project/support.py`: extraction and cleaning policy versions bumped to 11 and 7 to prevent reuse of old extraction results.
  - `corpus/pdf_native/objects.py`: bounded decompression and document resource budgets, validated predictor dimensions, bounded xref counts and EOF handling, and linear-time stream trimming.
  - `corpus/extraction.py`, `corpus/pdf_text_recovery.py`, and `generations/ingestion.py`: one source-scoped native-reader cache reused across staged page batches, with cleanup.
  - `corpus/pdf_native/extract.py`: column-aware recovery ordering.
  - `project/settings_document.py`: combined reranker-plus-threshold changes preserve `requires_ingest: true`; threshold-only rebuild costing was already correct.
- Separating columns before line merging when both columns have identical baselines remains in progress.
  - This may also affect `corpus/pdf_native/lines.py` or geometry helpers.
- Regression tests accompany these changes. Earlier test results do not validate the current combined tree.
- Do not run further validation while concurrent edits are being applied.
- Ingestion is stopped. Keep the previous generation selected and retain the unfinished checkpoint.
  - Do not restart ingestion or delete its checkpoint.
- Do not commit or push on behalf of this session. This restriction overrides the ordinary recording rules for this active work.
- Preserve concurrent edits, especially in `generations/ingestion.py`, `corpus/extraction.py`, `corpus/pdf_text_recovery.py`, and `corpus/pdf_native/lines.py`.

- Use `pathlib.Path`, type hints, and JSON-serializable payloads.
- A file under `evaluation/` changes only with the protocol it describes.
- Validate before committing, then push each change to `origin main` before starting the next.
  - Commit messages state what changed and why.
- End work with a clean tree: `git status --short` reports nothing.
  - Work that arrived in the tree from elsewhere counts as part of it. Commit it with this work, or state why not.

## Validation

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
- Source and retrieval integration tests use the real gateway, both indexes, hybrid and dense search, a known passage, and an excluded neighbour.
  - Restart offline and repeat a hybrid reranked search from caches.
- Offline unit tests cover:
  - RRF, failure atomicity, and no-op ingestion.
  - Source additions, changes, and removals, including byte changes that preserve size and mtime.
  - Forced regeneration, reuse, and final artifacts.
  - Reviewed metadata and exclusions across every read surface and filter, without rewriting generation files.
- Significant extraction changes also test a representative real collection without writing sources.
