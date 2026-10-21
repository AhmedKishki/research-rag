"""The one command line: one research project, in a terminal or a browser.

`research-rag` resolves a project, constructs a `ResearchService`, and calls it in
this process. `ui` serves the browser workspace and the agent surface from this
terminal, in the foreground, and there is no other way to bring an app up: a
command that starts one stays in the terminal that ran it, so Ctrl-C and a closed
window both reach it and nothing is left running that nobody is watching. One
console script reaches all of it, and the agent surface lives in `surfaces/mcp.py`
rather than here, so the terminal and the browser answer from the same call on the
same payload.

Retrieval loads its direct dependencies only when an operation needs them. A
command that reads local state starts no retrieval child process.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from collections.abc import Sequence
from contextlib import nullcontext
from typing import Any

from ...project.config import (
    CLI_COMMAND,
    ConfigurationError,
    ResearchConfig,
    apply_process_priority,
)
from ...project.policy import ResearchError
from ...project.settings import SETTINGS
from ...project.settings_document import describe_costs, describe_docs
from ...project.settings_layers import describe_settings
from ...runtime.control import ControlError, connect
from .corpus import _operate, _operations
from .credentials import _api_key
from .help import HELP_TOPICS, _command_usage, _help_menu
from .installation import _install, _update
from .parser import _parser
from .projects import _bridge_project_name, _config_kwargs, _init, _projects, _resolve
from .serving import (
    CommandResult,
    _bare_workspace,
    _clients,
    _disconnect,
    _service_processes,
    _start,
    _stop,
)


async def _doctor(args: argparse.Namespace, config: ResearchConfig) -> CommandResult:

    from ...runtime.doctor import mcp_entry_block, mcp_url_block, run_doctor

    if args.mcp_entry:
        # Two entries, because the app is reached two ways. The URL entry names
        # the port the app claimed, which exists only once it is up. The stdio
        # entry is complete either way, because the bridge starts the app itself.
        return CommandResult(
            text=mcp_url_block(config) + "\n" + mcp_entry_block(config)
        )
    entry_check = args.check_entry
    # A whole report is built from the same service call the `status` command
    # makes, so the two surfaces cannot disagree about this project. A check of
    # one entry file needs no project state, so it fetches none.
    status: dict[str, Any] = {}
    unavailable: str | None = None
    if entry_check is None:
        try:
            async with _operations(config) as operations:
                status = await operations.call("status")
        except ImportError as exc:
            # Without the retrieval stack there is no corpus answer to report, so
            # the report is told which of its checks that leaves unknown.
            unavailable = (
                f"{exc.name or exc} is not importable, so this project's corpus "
                "cannot be read. Install this project's dependencies to answer "
                "anything about its evidence."
            )
    result = run_doctor(
        config,
        status,
        running=_service_processes(config.project_root),
        entry=entry_check,
        prefetch=args.prefetch_models,
        repair=args.repair_runtime,
        status_unavailable=unavailable,
    )
    return CommandResult(text=result.text(), exit_code=result.exit_code)


# The named logger this process's own records travel on. A child logger of it
# (`research_rag.runtime.app`, `research_rag.core.service`) reaches the same
# handler; `logging.lastResort` is not used, so nothing prints unless this is set.
_LOGGER_NAME = "research_rag"


_LOG_LEVELS: dict[str, int] = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warn": logging.WARNING,
    "error": logging.ERROR,
}


def _configure_logging(config: ResearchConfig) -> None:
    """Let this process's own records reach stderr at the level the project set.

    The handler is attached to the named logger alone, so a dependency's DEBUG
    traffic stays silent and no root handler is installed or replaced. The
    handler is installed once, so a command that resolves logging twice does not
    double every line. This app never logs passage or query text: the records
    that become visible are reuse and compatibility decisions, which is why
    `runtime.log_level=debug` is the way to see why a build reused or rebuilt.
    """

    level = _LOG_LEVELS.get(str(config.settings.log_level).lower())
    if level is None:
        return
    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False
    if not any(
        getattr(handler, "_research_rag_stderr", False) for handler in logger.handlers
    ):
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        handler._research_rag_stderr = True  # type: ignore[attr-defined]
        logger.addHandler(handler)


async def _run(args: argparse.Namespace) -> CommandResult:

    if args.command == "projects":
        return CommandResult(payload=await _projects(args))
    if args.command == "init":
        return CommandResult(payload=_init(args))
    # These are about the installation or the account rather than a project, so they run
    # before one is resolved: resolving writes the project's portable state, and
    # a reader asking what is installed has no project yet.
    if args.command == "install":
        return CommandResult(payload=_install(args))
    if args.command == "update":
        return CommandResult(payload=await _update(args))
    if args.command == "api-key":
        return CommandResult(payload=_api_key(args))
    config = _resolve(args)
    apply_process_priority(config.nice)
    _configure_logging(config)
    if args.command == "config":
        if args.write:
            return CommandResult(payload=await _write_settings(args, config))
        print(
            describe_settings(
                SETTINGS, config.settings.as_values(), config.settings_provenance
            )
        )
        print(describe_docs())
        print(describe_costs(config.settings))
        return CommandResult()
    if args.command == "start":
        return await _start(args, config)
    if args.command == "clients":
        return CommandResult(payload=_clients(config))
    if args.command == "lan":
        with nullcontext(connect(config)) as control:
            if control is None:
                raise ResearchError("No app is running; run research-rag start.")
            return CommandResult(payload=control.set_lan(args.action == "enable"))
    if args.command == "disconnect":
        return CommandResult(payload=_disconnect(args, config))
    if args.command == "stop":
        return CommandResult(payload=_stop(args, config))
    if args.command == "doctor":
        return await _doctor(args, config)
    async with _operations(config) as operations:
        payload = await _operate(args, operations)
    return CommandResult(payload=payload)


async def _write_settings(
    args: argparse.Namespace, config: ResearchConfig
) -> dict[str, Any]:
    """Write `--write` values through the service call the workspace makes.

    A running app answers through the control API and takes the values in
    memory; without one, the write runs in process. Either way the revision read
    here guards the write, so a change made elsewhere in between is refused.
    """

    values: dict[str, str] = {}
    for item in args.write:
        key, separator, value = item.partition("=")
        if not separator or not key.strip():
            raise ResearchError(f"--write takes KEY=VALUE, not {item!r}")
        values[key.strip()] = value
    async with _operations(config) as operations:
        current = await operations.call("settings_read")
        return await operations.call(
            "settings_write",
            values=values,
            expected_revision=current["revision"],
            confirm=args.confirm,
        )


def main(argv: Sequence[str] | None = None) -> None:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.version:
        # Before the project is resolved, like `help`: a reader asking what is
        # installed has no project yet, and the answer must not fail on a
        # directory that happens to hold one.
        from ...core.version import version_lines

        sys.stdout.write("\n".join(version_lines()) + "\n")
        return
    if args.command is None:
        # No command is the one thing a reader types without thinking, so it opens
        # the workspace rather than printing a menu: the workspace is what every
        # other command in this app exists to reach, and the app it opens stays in
        # this terminal, so closing the terminal closes it. Before the loop
        # starts, because this call owns one.
        try:
            result = _bare_workspace(args)
        except (
            ConfigurationError,
            ControlError,
            ResearchError,
            OSError,
            ValueError,
        ) as exc:
            raise SystemExit(f"{CLI_COMMAND}: {exc}") from exc
        # A takeover that was refused is not a served workspace.
        if result.exit_code:
            raise SystemExit(result.exit_code)
        return
    if args.command == "help":
        # Before the project is resolved: the menu is what a reader has
        # precisely when they have no project yet, and it must not fail on a
        # directory that holds one.
        if args.topic is None:
            sys.stdout.write(_help_menu())
        elif args.topic in HELP_TOPICS:
            sys.stdout.write(HELP_TOPICS[args.topic])
        else:
            sys.stdout.write(_command_usage(parser, args.topic))
        return
    if args.command == "mcp":
        # The bridge owns stdio and its own event loop, so it runs before this
        # process starts one. It is handed a project's name, because a client
        # entry that named a directory could not be copied to another machine.
        try:
            project_name = _bridge_project_name(args)
        except (ConfigurationError, ResearchError) as exc:
            raise SystemExit(f"{CLI_COMMAND}: {exc}") from exc
        from .. import bridge

        bridge.main(project_name, settings=_config_kwargs(args), name=args.client_name)
        return
    try:
        result = asyncio.run(_run(args))
    except (
        ConfigurationError,
        ControlError,
        ResearchError,
        OSError,
        ValueError,
    ) as exc:
        raise SystemExit(f"{CLI_COMMAND}: {exc}") from exc
    if result.text is not None:
        sys.stdout.write(result.text)
    elif result.payload is not None:
        print(json.dumps(result.payload, ensure_ascii=False, indent=2))
    if result.exit_code:
        raise SystemExit(result.exit_code)


if __name__ == "__main__":
    main()
