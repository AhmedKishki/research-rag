"""Retired `research-rag-runtime` shim.

The runtime that installed and validated a pinned UltraRAG source snapshot is
removed. Normal retrieval answers in process and needs no external runtime.
This entry point accepts the historical flags so an old invocation succeeds
with a clear notice instead of a missing executable. It downloads nothing,
validates nothing, reads and writes no cache, and starts no process.
"""

from __future__ import annotations

import argparse
from pathlib import Path

RETIREMENT_MESSAGE = """\
No external runtime is required.

research-rag serves retrieval in process, so research-rag-runtime no longer
installs or validates an UltraRAG snapshot and nothing needs to be prepared
ahead of a run. 'doctor --repair-runtime' has no runtime left to repair.
"""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="research-rag-runtime",
        description=(
            "Retired. No external runtime is required. This command accepts the "
            "historical flags, prints a notice, and exits without installing or "
            "validating anything."
        ),
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=None,
        help="Retired. No runtime cache is read or written.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Retired. Nothing is ever downloaded.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Retired. There is no runtime to validate.",
    )
    return parser


def main() -> None:
    _parser().parse_args()
    print(RETIREMENT_MESSAGE)


if __name__ == "__main__":
    main()
