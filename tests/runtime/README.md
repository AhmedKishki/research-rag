# The serving process, its health report, and the commands that need no project

These tests hold the one process a project runs on, the control API the terminal holds on it, the attachment rule that makes the terminal own the app's lifetime, the four-state health report, the `doctor` reader over the same report, the desktop-entry and console-command installation, and the release comparison that decides whether an update applies. The gateway is faked or left closed, so no test here starts it.

- `test_app.py` — a real app on a real loopback port: the agent endpoint answers a session once mounted, the workspace and the agent surface share one service, a control settings write is refused from another site and without a JSON body, a client is listed once and can be disconnected, and a dropped session never reaches the tools.
- `test_attached_workspace.py` — the pid, port, and tty records, a bare call that never starts a second app, a detached app named together with its remedy, and the stop sweep that signals only a process this project can prove it owns.
- `test_health.py` — one check per condition in `ok`, `warn`, `blocked`, or `unknown`: a mismatched runtime, a missing model, a held project lock, a stale lock left by a dead process, free space measured against the build it has to fit, an app older than the installed code, and a check that did not run never reading as healthy.
- `test_doctor.py` — a default run that reports every check and changes nothing, the two operations that reach the network tested against stand-ins, and each desktop entry checked for the project it names, the bridge it runs, and the timeout it declares.
- `test_installation.py` — `install`, `uninstall`, and `--force`: a foreign command or entry this app did not write is reported and left alone unless forced, an entry naming one project is replaced by one serving every project, and a path with spaces survives the round trip through the entry.
- `test_cli.py` — `surfaces/cli.py`: command routing, project selection by name, the `projects` listing and its missing directories, `init` in its three forms, exclusion and metadata writes, the gateway opened by the first call and reused, a reading command that never opens it, and the generation commands with their repeated-id confirmation.
- `test_release.py` — the release a tag may carry, read through `git` against a temporary repository and a local bare remote, so no test reaches a network.
- `test_update.py` — the release decides and the branch head is reported beside it: a dirty tree refuses and names the files, an unreachable remote is an answer rather than a failure, the tool that installed the distribution is the one that updates it, and a project lock held by a build refuses the whole update.
- `test_version.py` — the running and installed versions, the `--version` answer, and that the declared version matches the installed distribution.
- `test_update_script.py` — `scripts/update.sh`, which is not a module: it delegates every flag it keeps to `research-rag update` and rejects an unknown option.

`test_cli.py` and parts of `test_installation.py` and `test_update.py` exercise `src/research_rag/surfaces/cli.py` from this folder, because what they hold is the process the command reaches rather than the surface itself. `install_entry.py` and `tool_ownership.py` are reached through `installation` and `update`, which re-export them.

## Running these

```bash
.venv/bin/python -m pytest tests/runtime -q
```

A change to the app's lifetime, the control API, a health state, or an installation path belongs here, because none of those is reachable from a unit test with a fake transport. A change to an answer shape does not, and belongs to `tests/core`.

## What it mirrors

`src/research_rag/runtime/`, and the command line that reaches it.