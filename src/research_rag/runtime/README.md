# `runtime`

The running process, and the commands that manage it.

| Module | Holds |
|---|---|
| `app.py` | `App`: the one instance that serves a port, holds the client registry, and composes the surfaces |
| `control.py` | The loopback control API, and the synchronous handle the command line speaks through it |
| `network.py` | Home-LAN interface discovery and workspace host, peer, origin, and route authorization |
| `process.py` | How a child process is asked to run and what a failure of it says |
| `health.py` | The checks one health report is made of, and what each one blocks |
| `doctor.py` | What is wrong with an installation, and the repair each fault has |
| `update.py` | Whether this installation is behind a release, and the plan that moves it |
| `tool_ownership.py` | Which tool installed this app, and the command that upgrades it |
| `installation.py` | The desktop entry and the icon |
| `install_entry.py` | The console script on an account's `PATH` |
| `release.py` | What a remote publishes as a release, read from its git tags |
| `version.py` | The version the running process started with, and the one installed now |

## Rules

- An app belongs to the terminal that started it. Nothing here starts one to answer
  a question, and `stop` asks an app to stop rather than killing it.
- `app.py` and `control.py` import each other, and that cycle is load-bearing: the routes need an `App`, and the `App` needs its routes.
- A command that answers must work on a machine where the retrieval stack is
  absent. `import research_rag.runtime.update` is the test of it.
- `process.py` is the only place a child process is run. A second `subprocess`
  call is a second way for a command to fail without saying why.
