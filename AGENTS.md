---
name: AGENTS.md
description: Research app boundaries, evidence contracts, and validation rules.
---

# AGENTS.md

## Scope

- Project-scoped PDF/EPUB evidence retrieval, not answer generation.
- One process and loopback port per project, shared by the workspace, MCP, and CLI.
- Ship the gateway, workspace, and settings layer here; do not depend on a sibling project.

## Documentation responsibilities

- Give each fact one owner; link rather than repeat it.

| File | Owns |
|---|---|
| `README.md` | The quickstart: requirements, installation, a first project, a first search, the three front ends, and the limits a first reader hits. |
| `help` in `surfaces/cli.py` | The canonical command reference: `HELP_GROUPS` for every command, `HELP_TOPICS` for a subject, and each command's own usage. |
| `STORAGE.md` | Files, fields, portability, and hand edits. |
| `FEATURES.md` | Shipped, excluded, and planned capabilities; UltraRAG reuse. |
| `MEASUREMENTS.md` | Measurement protocols, operating envelope, and tunable mechanisms. Read constants, models, revisions, and thresholds from `default.toml` and code. |
| `ROADMAP.md` / `TODO.md` | deferred ideas / open work. |
| `AGENTS.md` | Engineering constraints. |
| Module docstrings | Local mechanisms and boundaries. |

- The README points at the command reference instead of repeating it.
  - Its examples parse against the parser; `tests/gates/test_documentation.py` holds that.
  - A command added to the parser must reach `HELP_GROUPS` in the same change, which the same gate holds.
- Markdown states current rules, not review logs, completed tasks, or superseded decisions.
  - Remove finished TODOs and superseded measurements; Git retains history.
- Re-measure changes to corpus, extraction, or retrieval defaults and update every affected figure.
- Reference file paths or harness names, not drifting section numbers.
- Keep the child README standalone, without sibling comparisons or links.
- Use active verbs and concrete nouns, with one fact per sentence.
  - Cut preambles, hedges, sales language, repeated headings, and filler such as *very*, *really*, *actually*, *simply*, *just*, *importantly*, and *of course*.
  - Use bullets for lists; delete sentences that add no rule or fact.
- Documentation and hand-edit gates: `tests/gates/test_documentation.py`, `tests/core/test_review_state_edits.py`.

## Presenting decisions to the user

- Present numbered, concrete options with identifiers, smallest change, cost, risk, research-contract effects, and reversibility.
- Recommend one option with a reason; include "no change" when work can proceed without a choice.
- Obtain a choice before changing generation artifacts, retrievable evidence, portable state, or on-disk layouts.

## Current baseline

- `pyproject.toml` owns the version and Python range; runtime version reads installed distribution metadata.
  - `update` compares published releases, not branch heads.
  - Show the matching GitHub changelog before approval; declining or unavailable approval must not install, stop apps, or change project state.
  - `tests/runtime/test_release.py` validates release versions and local tag targets.
- `LICENSE` grants Apache-2.0 for this code; `NOTICE` owns separate dependency terms, including AGPL-3.0 extraction.
- `README.md` owns the quickstart; `research-rag help` owns the command reference.
  - `install` writes PATH and desktop entries, not a serving process.
  - `install`, `update`, `help`, and `--version` need no project at all.
  - `help` groups commands and delegates command-specific usage.
  - `--version` and `update` share app/installed/restart reporting.
- Preserve `USER_CONFIG_DIRECTORY = research-rag`, `SETTINGS_ENVIRONMENT_PREFIX = RESEARCH_RAG_`, and the `research-rag` model cache.
  - Do not use these names for another product; see `tests/project/test_data_roots.py`.
- Pins live in `pyproject.toml` and `gateway/manifest.py`.
  - Keep the `bm25s` non-ASCII stopword fork until upstream fixes round-tripping; reject invalid lists promptly.
  - `_pdf_locator` falls back to the physical page on `Page.get_label()`'s `IndexError`; narrow the guard when upstream issue 5140 closes.

## Rules

### The contract a reader relies on

- Preserve prominent UltraRAG credit, upstream links, licences, `NOTICE`, and the independent-project disclaimer.
  - Credit THUNLP, NEUIR, OpenBMB, AI9stars, and contributors without implying endorsement.
- No original source document is edited or written: `sources/` is the authority for a quotation.
- `text` in a result is cleaned semantic text and never a transcript.
- `direct_quote_safe` is `false` on every passage the app builds, and that flag is the only place the rule appears per passage.
- An agent gets eight tools and one resource with bounded answers, never a corpus inventory.
  - The workspace and CLI `sources` own inventory access.
- A generation the app cannot serve is stated as `hybrid_ready: false` beside `generation_upgrade_required`, and it leaves the old one searchable with a warning.
- Expose only caller decisions, not engine tuning.
  - Agents get no retrieval mode, output view, or chunk tuning; CLI `search --method` and `--no-rerank` are diagnostic exceptions.
- CLI and agent `status` share bounded `tool_views` projections.
  - Allowlist actionable app fields; never merge process or client inventories into MCP answers.
- Client entries and generated templates carry a project name, never a path.
  - `mcp --project-name` or `RESEARCH_RAG_PROJECT_NAME` resolves exactly through `registry.py`.
  - Refuse `mcp --project-root` and `mcp --project`.
- For an uninitialized project, open the session with only `status`, an `init` remedy in `blocked_by`, and no corpus tools.
  - `tests/surfaces/test_bridge.py` covers this surface in `surfaces/mcp.py`.
- Generation of an answer is out of scope. `ROADMAP.md` holds it and the API, citation contract, and tests it would need.
- No bundle operations exist: `export_bundle` and `import_bundle` are not available.

### The state a reader must be able to trust

- One app process serves each configured project root, and every surface shares that process, its project lock, and its single UltraRAG gateway.
  - Implement capabilities once in `ResearchService`; running CLI commands use the control API.
  - Without an app, answer local-state operations in process.
- Keep project artifacts under `<project>/.research-rag`, including namespaced pid, port, and lock files.
  - `STORAGE.md` owns runtime relocation and its marker.
  - Share only immutable model binaries through the user cache, never corpus, index, log, or query state.
- Preserve deterministic IDs, normalized source-relative paths, and locators.
  - `source_id` survives byte changes; `document_id` identifies path plus content version.
  - Chunk IDs may change with text or chunking; document/chunk IDs need not survive incompatible generations.
- Activate `current.json` only after complete indexing and validation of both BM25 and dense artifacts in the same generation.
- Honor hand-edited review state; reject unknown fields, wrong types, and non-normalized paths explicitly.
- Apply reviewed metadata and passage exclusions at read time, without rewriting artifacts or marking them `stale`.
- Source and passage exclusions remain explicit, reversible, project-local, and consistent across surfaces.
  - Do not guess duplicates automatically.
  - Report unmatched passage exclusions in `status`; `requires` must not prescribe `ingest` to recover a removed chunk.
- This app alone owns portable `project.json`, `source-metadata.json`, `source-exclusions.json`, and `source-catalog.json`.
  - Keep passage decisions separate in `chunk-exclusions.json`.
  - Write no machine-specific portable state except the relocated-runtime record.
- `registry.py` holds project id, name, and root, never cached project state or cross-project permissions.
  - Resolve names exactly; refuse ambiguity and never resolve a name through an id.
- Own `settings_layers`, `SETTINGS`, `EffectiveSettings`, `default.toml`, and layer-location names here.
  - Settings writes merge only into `<project>/.research-rag/config.toml`.
  - Refuse writes to keys overridden by higher layers.
  - Determine rebuild cost from generation records, not `Setting.layer`.
- Keep schema/policy versions, retrieval-method sets, and boundary names in code.
  - Artifact-affecting tunables enter retrieval fingerprints or recorded chunk settings; other tunables are runtime settings.
- Validate generation names against the builder's pattern before forming paths; lock `use_generation` and `remove_generation`.
  - Refuse operations on the generation being searched, and removal of one pending activation.
  - Require the removal id twice; preserve open-reader safety.
- Make artifacts durable before commit points, with directory fsyncs grouped per unit.
  - Only disposable peer handoffs rewritten before every use may skip fsync.
  - Checkpoint bounded batches, not whole phases.
- Allow one ingestion build per project; promptly reject contention and name the resident phase rather than queue.
- Reuse extraction only when bytes, storage policy, and processing fingerprints match.
  - Vector reuse also requires exact canonical text, model revision, and dimension.
  - `force_recompute=True` disables all reuse.
- Dispatch dense backends from the manifest, never substitute one during reuse.
- Extraction failure for a selected source aborts the new generation and preserves the current one.

### Boundaries a contributor must not cross

- Keep the engine free of Starlette, FastMCP, and surface imports; surfaces must not import or redeclare one another.
  - `tests/gates/test_architecture.py` enforces the split.
- Ship the workspace in `surfaces/workspace/` with its adapter, source authorization, and host.
  - Hide unsupported capabilities; `ResearchUIAdapter._arguments` removes unsupported arguments.
  - Call `ResearchService`, never read generation artifacts directly; allowlisted originals are the sole file-serving exception.
- Browser and control writes must be same-origin, JSON-only, and loopback-only, without caller-selected output paths.
  - Remote exposure requires an explicit security design.
- Ingest only regular PDF/EPUB files beneath the configured source root; reject symlinks and traversal.
- Do not patch UltraRAG, expose vanilla operations, or move dense backends into the gateway.
- Preserve CLI access to every workspace capability.
  - Settings writes belong to the workspace, not agents; `research-rag config` is read-only.
  - Open a browser only with `--start-ui`.
- Use `child_process_environment()`, never `dict(os.environ)`, for subprocesses.
- Never point the app at a second copy of itself.
- Signal only proven project-owned processes; require the app entrypoint and matching `--project-root` in stop sweeps.
  - Never signal the sweep itself.
- Serve from the starting terminal; report, rather than adopt, a detached serving process.
  - Closing the starting terminal ends its app; only `stop` ends a detached app.
  - Read terminal ownership from `/proc/<pid>/stat`, not merely `research-rag-ui.tty`.
  - `start`, bare invocation, `projects`, and `stop` report the condition with a `stop` remedy, never launch a second app.
  - Detached serving remains deferred in `TODO.md`; `tests/runtime/test_attached_workspace.py` covers it.
- Keep private projects, notes, drafts, and original PDFs/EPUBs out of Git; add new personal paths to `.gitignore`.
- State unimplemented limits explicitly, never as shipped capabilities.

### Reporting a condition

- Name actionable conditions with a reason, pasteable remedy, and relevant log path.
  - Never report only `Connection closed`.
- `health.py` owns read-only checks: `ok`, `warn`, `blocked`, or `unknown` when unchecked.
  - Start no process, network request, or write while building health reports.
  - Omit empty `blocked_by` and `degraded`; healthy projects need neither.
- Default `doctor` reads the same report without writing.
  - Only explicit `--prefetch-models` and `--repair-runtime` may fetch or write; quarantine discarded evidence.
- Diagnose gateway failures through the requesting operation and component logs, not process-tree inspection.
- Resolve settings and serve before opening the gateway; local reads must work without a managed runtime.
  - Lightweight commands must not require the retrieval stack; `TODO.md` owns remaining dependency cleanup.
- Move blocking extraction and filesystem scans off the event loop.
- Serialize project operations with both the in-process lock and cross-process `project.lock`.

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

### What each layer owns

- Each source folder owns one concern, with its boundaries in its README and tests in the matching test folder.
- Import within the module's folder or from the folder above, not sideways between sibling folders.
- Keep paths used by tests in their owning tests; update them when renaming folders.
- Import shared concerns rather than copy them, including state names in `project/state_files.py` and shared surface contracts.
- Error types, schema versions, and boundary names must be importable without fastembed.

## Working rule

- Use `pathlib.Path`, type hints, and JSON-serializable payloads.
- A file under `evaluation/` changes only with the protocol that file describes.
- Validate before committing, then push each change to `origin main` before starting the next.
  - Commit messages state what changed and why.
- The work ends with a clean tree: `git status --short` reports nothing.
  - Work that arrived in the tree from elsewhere counts as part of the tree, so it is committed with this work or the reason it is not is stated.
  - The collection's submodule pointer is pushed in its own commit after the child's, never in the same one.

## Validation

```bash
uv lock --check
uv run ruff format --check .
uv run ruff check .
uv run pytest -q
uv run python -m compileall -q src tests
```

- A retrieval-quality claim is produced by `uv run python scripts/evaluate_retrieval.py --project <project> --offline`, with `--validate-only` added first.
  - Record valid measurements in `MEASUREMENTS.md`; never change documented defaults on an unrecorded run.
- Workspace changes test safe source resolution, argument forwarding, and the real host without source mutations.
  - Run the app's `surfaces/workspace/` tests.
- Source/retrieval integration tests use the real gateway, both indexes, hybrid/dense search, a known passage, and an excluded neighbor.
  - Restart offline and repeat hybrid reranked search from caches.
- Offline unit tests cover RRF, failure atomicity, no-op ingestion, source additions/changes/removals, preserved-size/mtime byte changes, forced regeneration, reuse, and final artifacts.
  - Test reviewed metadata/exclusions across every read surface and filter without rewriting generation files.
- Significant extraction changes also test a representative real collection without writing sources.
