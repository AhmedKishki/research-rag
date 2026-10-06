# Runtime tests

The serving process, its health report, and the commands that need no project. The gateway is faked or left closed.

- `test_app.py`: a real app on a loopback port. The agent endpoint answers once mounted, the workspace and agent surface share one service, a control settings write is refused from another site or without a JSON body, a client is listed once and can be disconnected, and a dropped session never reaches the tools.
- `test_attached_workspace.py`: the pid, port, and tty records, a bare call that never starts a second app, a detached app named with its remedy, a process with no terminal refused, an app stopped when its terminal is lost, and a stop sweep that signals only a provably owned process.
- `test_health.py`: one check per condition in `ok`, `warn`, `blocked`, or `unknown`.
  - A mismatched runtime, a missing model, a held project lock, and a stale lock from a dead process.
  - Free space against the build it must fit, and an app older than the installed code.
  - A check that did not run never reads as healthy.
- `test_doctor.py`: a default run reports every check and changes nothing. The two network operations run against stand-ins. Each desktop entry is checked for its project, bridge, and timeout.
- `test_installation.py`: `install`, `uninstall`, and `--force`. A foreign command or entry is left alone unless forced, an entry for one project is replaced by one serving every project, and a path with spaces survives.
- `test_cli.py`: command routing, project selection by name, the `projects` listing, `init` in its three forms, exclusion and metadata writes, the gateway opened by the first call and reused, a reading command that never opens it, and the generation commands with their repeated-id confirmation.
- `test_release.py`: the release a tag may carry, read through `git` against a temporary repository and a local bare remote.
- `test_update.py`: the release decides and the branch head is reported beside it. A dirty tree refuses and names the files, an unreachable remote is an answer, the installing tool is the one that updates, and a held project lock refuses the update.
- `test_version.py`: running and installed versions, the `--version` answer, and the declared version matching the installed distribution.
- `test_update_script.py`: `scripts/update.sh` delegates every flag it keeps to `research-rag update` and rejects an unknown option.

- `test_cli.py` and parts of `test_installation.py` and `test_update.py` exercise `surfaces/cli.py` here, because they hold the process the command reaches.
- `install_entry.py` and `tool_ownership.py` are reached through `installation` and `update`, which re-export them.
- A change to an answer shape belongs in `tests/core`.
