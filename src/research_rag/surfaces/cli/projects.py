"""Which project a command is about: the global options that name one, the
registered projects to choose from, and `init`, which creates one.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from ...project.config import (
    CLI_COMMAND,
    ConfigurationError,
    ResearchConfig,
    configured_source_directory,
    project_command,
    resolve_config,
)
from ...project.policy import ResearchError
from ...project.registry import register as register_project
from ...project.registry import registry_path
from ...project.registry import resolve as resolve_registered
from ...runtime.account import account_projects

# The variable a client entry that cannot pass an argument sets instead. It is
# the counterpart of `RESEARCH_RAG_PROJECT_ROOT`, and it exists for the same
# reason: a client that offers only an environment block still has to name a
# project.
PROJECT_NAME_ENV = "RESEARCH_RAG_PROJECT_NAME"


# What a project's own .gitignore keeps out of version control: the derived
# state that can be rebuilt. The descriptor, catalogs, and review files are
# small, portable, and worth keeping.
VERSION_CONTROL_NOTES = (".research-rag/runtime/",)


def _project_path(args: argparse.Namespace) -> Path:
    """Return the project this call names, by registry selector or by path.

    `--project` and `--project-root` are two ways to say the same thing, so a
    call that supplies both is refused rather than resolved by precedence.
    """

    selector = getattr(args, "project", None)
    named_root = getattr(args, "project_root", None)
    if selector and named_root:
        raise ConfigurationError(
            f"--project {selector!r} and --project-root {str(named_root)!r} name "
            "two projects; pass one."
        )
    if selector:
        return resolve_registered(selector).project_root
    return Path(named_root or ".").expanduser().resolve()


def _config_kwargs(args: argparse.Namespace) -> dict[str, Any]:

    return {
        "runtime_root": args.runtime_root,
        "model_cache_root": args.model_cache_root,
        "offline": args.offline,
        "dense_backend": args.dense_backend,
        "embedding_threads": args.embedding_threads,
        "reranker_model": args.reranker_model,
        "config_path": args.config,
        "settings_overrides": args.set_overrides,
    }


def _resolve(args: argparse.Namespace) -> ResearchConfig:
    """Resolve an existing project, reusing the source directory it recorded."""

    named = getattr(args, "project", None) or getattr(args, "project_root", None)
    project = (
        _only_project_or_ask()
        if args.command == "start" and not named
        else _project_path(args)
    )
    return resolve_config(
        project,
        source_directory=configured_source_directory(project),
        **_config_kwargs(args),
    )


def _init(args: argparse.Namespace) -> dict[str, Any]:
    """Create a project, or attach the portable state to a directory that exists.

    A project *is* a directory with `.research-rag` beside whatever is already
    there, and no file the user wrote is touched.
    """

    project = _project_path(args)
    created: list[str] = []
    if project.exists() and not project.is_dir():
        raise ConfigurationError(f"Project root is not a directory: {project}")
    if not project.exists():
        project.mkdir(parents=True)
        created.append("project_root")
    # An existing project records its own source directory, and a project that is
    # not there yet has no descriptor to read, so both settle before the
    # descriptor is written.
    sources = args.sources or configured_source_directory(project)
    config = resolve_config(
        project,
        source_directory=sources,
        project_name=args.name,
        **_config_kwargs(args),
    )
    if not config.source_root.exists():
        config.source_root.mkdir(parents=True)
        created.append("source_root")
    try:
        registered = register_project(
            config.project_id, config.project_name, config.project_root
        )
    except (OSError, ResearchError) as exc:
        raise ConfigurationError(
            f"The project at {config.project_root} was created, but this "
            f"installation could not record it ({exc}). Keep addressing it with "
            "--project-root until the record at "
            f"{registry_path()} can be written."
        ) from exc
    return {
        "status": "ready",
        "project_root": str(config.project_root),
        "project_id": config.project_id,
        "project_name": config.project_name,
        "source_root": str(config.source_root),
        "created": created,
        "registered": registered.as_record(),
        "registry_path": str(registry_path()),
        "keep_out_of_version_control": list(VERSION_CONTROL_NOTES),
        "next_steps": [
            f"Add PDF, EPUB or MOBI sources to {config.source_root}.",
            project_command(config.project_root, "ingest"),
            project_command(config.project_root, "search", "your question"),
            project_command(config.project_root, "start"),
        ],
    }


async def _projects(args: argparse.Namespace) -> dict[str, Any]:
    """Report every registered project, and whether an app is serving it.

    The listing is the account's own record read once, so the workspace's
    selector names the same projects this command does. A project whose app is
    not running is reported as such rather than started, because a listing is a
    question about what exists.
    """

    return account_projects()


def _only_project_or_ask() -> Path:
    """Choose a registered serving project without consulting the working directory."""

    registered = account_projects()["projects"]
    if not registered:
        raise ConfigurationError(
            "No project is initialised on this machine, so there is no workspace "
            f"to open. Create one with '{CLI_COMMAND} init --project-root DIR --name "
            "NAME'."
        )
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        import shlex

        raise ConfigurationError(
            "Choosing a registered project requires an interactive terminal. "
            "Name one explicitly: "
            + "; ".join(
                f"{CLI_COMMAND} --project {shlex.quote(str(entry['project_name']))} start"
                for entry in registered
            )
            + "."
        )
    return Path(str(registered[_pick_registered_project(registered)]["project_root"]))


def _pick_registered_project(registered: list[dict[str, Any]]) -> int:
    """Read terminal arrow keys, restoring its attributes even after cancellation."""

    import os
    import select
    import termios
    import tty

    fd = sys.stdin.fileno()
    original = termios.tcgetattr(fd)
    chosen = 0

    def draw() -> None:
        entry = registered[chosen]
        name = "".join(c for c in str(entry["project_name"]) if c.isprintable())
        app = entry.get("app", {})
        state = " — already served" if app.get("running") else ""
        if app.get("detached"):
            state = " — served detached"
        sys.stdout.write(
            f"\r\033[2KProject {chosen + 1}/{len(registered)}: {name}{state}"
        )
        sys.stdout.flush()

    try:
        tty.setraw(fd)
        sys.stdout.write("Choose project: Up/Down, Enter; Escape/Ctrl-C cancels\r\n")
        draw()
        while True:
            key = os.read(fd, 1)
            if key in {b"\r", b"\n"}:
                return chosen
            if key in {b"", b"\x03"}:
                break
            if key != b"\x1b":
                continue
            sequence = b""
            for _ in range(2):
                if not select.select([fd], [], [], 0.15)[0]:
                    break
                sequence += os.read(fd, 1)
            if sequence in {b"[A", b"OA"}:
                chosen = (chosen - 1) % len(registered)
            elif sequence in {b"[B", b"OB"}:
                chosen = (chosen + 1) % len(registered)
            else:
                break
            draw()
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, original)
        sys.stdout.write("\n")
        sys.stdout.flush()
    raise ConfigurationError("No project chosen, so nothing was started.")


def _bridge_project_name(args: argparse.Namespace) -> str:
    """Return the project this MCP entry names, and refuse the ones it may not.

    A client entry is copied between machines, and a path in one is true on the
    machine it was written on. The name is resolved here, by this installation,
    so the entry itself names a project and nothing about where it lives.
    """

    for option, value in (
        ("--project-root", getattr(args, "project_root", None)),
        ("--project", getattr(args, "project", None)),
    ):
        if value:
            raise ConfigurationError(
                f"The MCP entry names a project, not a directory: drop {option} "
                "and pass --project-name NAME, which this installation resolves to "
                "the directory the project was initialised at."
            )
    name = getattr(args, "project_name", None)
    if not name or not name.strip():
        raise ConfigurationError(
            "The MCP entry needs --project-name NAME, or "
            f"{PROJECT_NAME_ENV} in the client's environment, naming a project "
            "this installation initialised."
        )
    return name.strip()
