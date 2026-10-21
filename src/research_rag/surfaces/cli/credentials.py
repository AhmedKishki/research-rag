"""`api-key`: the account's DeepSeek key, set, cleared, or reported.

The key never reaches an argument, so it never lands in shell history or a
process listing, and no answer here prints it.
"""

from __future__ import annotations

import argparse
import getpass
import sys
from typing import Any

from ...project.credentials import (
    clear_deepseek_key,
    credentials_path,
    deepseek_key,
    set_deepseek_key,
)


def _api_key(args: argparse.Namespace) -> dict[str, Any]:
    if args.action == "set":
        set_deepseek_key(_read_key())
    elif args.action == "clear":
        clear_deepseek_key()
    key, origin = deepseek_key()
    return {
        "deepseek": {
            "set": key is not None,
            "origin": origin,
            "path": str(credentials_path()),
        }
    }


def _read_key() -> str:
    if sys.stdin.isatty():
        return getpass.getpass("DeepSeek API key: ")
    return sys.stdin.readline()
