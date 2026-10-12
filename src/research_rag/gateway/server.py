"""Retired `research-rag-gateway` shim.

The stdio gateway that proxied UltraRAG's own MCP servers is removed. This
entry point keeps the historical command name so an old wrapper or client
fails with guidance instead of a missing executable. It accepts the flags the
command used to take, starts no process, imports no upstream library, reads no
cache, and writes no file.
"""

from __future__ import annotations

import argparse
import sys

RETIREMENT_MESSAGE = """\
research-rag-gateway has been retired.

This app no longer proxies UltraRAG's MCP servers, so there is no gateway to
start, no UltraRAG runtime to install, and no optional 'legacy' dependency to
add. The app serves its own eight evidence-retrieval tools itself.

Use the app's own agent surface instead:

    research-rag mcp --project-name <name>

Run 'research-rag help' for the command reference, or
'research-rag help mcp' for the agent surface.
"""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="research-rag-gateway",
        description=(
            "Retired. This command no longer starts a gateway. It accepts the "
            "historical flags, prints migration guidance, and exits without "
            "starting anything. Use 'research-rag mcp --project-name <name>'."
        ),
    )
    parser.add_argument(
        "--namespace",
        action="append",
        help="Retired. The app's own tools replace every UltraRAG namespace.",
    )
    parser.add_argument(
        "--ultrarag-root",
        help="Retired. No UltraRAG checkout is read.",
    )
    parser.add_argument(
        "--runtime-cache-root",
        help="Retired. No runtime cache is read or written.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Retired. The app never downloads a gateway runtime.",
    )
    parser.add_argument(
        "--workspace-root",
        help="Retired. No external working directory is used.",
    )
    parser.add_argument(
        "--python",
        dest="python_executable",
        default=sys.executable,
        help="Retired. No child interpreter is started.",
    )
    parser.add_argument(
        "--log-level",
        choices=("debug", "info", "warn", "error"),
        default="warn",
        help="Retired. No child process writes a log.",
    )
    return parser


def main() -> None:
    # Parse so a historical invocation is recognized, then refuse clearly. The
    # parse produces no side effect: --help prints this retirement and exits 0.
    _parser().parse_args()
    raise SystemExit(RETIREMENT_MESSAGE)


if __name__ == "__main__":
    main()
