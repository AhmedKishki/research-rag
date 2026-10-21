"""The account's DeepSeek key: a credential, never a setting.

A setting travels with a project and is printed by `config`; a key must do
neither. It lives in `credentials.json` beside the account's `config.toml`,
written readable by its owner alone, or in `RESEARCH_RAG_DEEPSEEK_API_KEY`,
which wins. It is read on every call, so a running app takes a new key at its
next answer. `STORAGE.md` owns the file's fields.

No message here repeats a key: a refusal names the file and the field.
"""

from __future__ import annotations

import json
import os
import re
import stat
from pathlib import Path
from typing import Any

from ..storage.durable_io import atomic_file
from .policy import ResearchError
from .settings import USER_CONFIG_DIRECTORY
from .settings_layers import user_config_path

CREDENTIALS_FILE = "credentials.json"
CREDENTIALS_SCHEMA_VERSION = 1
DEEPSEEK_KEY_ENVIRONMENT = "RESEARCH_RAG_DEEPSEEK_API_KEY"
ORIGIN_ENVIRONMENT = "environment"
ORIGIN_FILE = "account file"

# A bearer token is printable ASCII without spaces.
_KEY = re.compile(r"[\x21-\x7e]+")
_PROVIDER_FIELDS = {"deepseek": frozenset({"api_key"})}


def credentials_path() -> Path:
    """The account's credentials file, beside its `config.toml`."""

    return user_config_path(USER_CONFIG_DIRECTORY).with_name(CREDENTIALS_FILE)


def deepseek_key() -> tuple[str | None, str | None]:
    """Return the key in force and where it came from, or `(None, None)`."""

    environment = os.environ.get(DEEPSEEK_KEY_ENVIRONMENT, "").strip()
    if environment:
        return environment, ORIGIN_ENVIRONMENT
    key = _read().get("deepseek", {}).get("api_key")
    return (key, ORIGIN_FILE) if key else (None, None)


def set_deepseek_key(key: str) -> None:
    key = key.strip()
    if not _KEY.fullmatch(key):
        raise ResearchError(
            "A DeepSeek API key is one word of printable characters; nothing was "
            "written."
        )
    document = _read()
    document["deepseek"] = {"api_key": key}
    _write(document)


def clear_deepseek_key() -> None:
    if not credentials_path().exists():
        return
    document = _read()
    document["deepseek"] = {}
    _write(document)


def credentials_exposed() -> bool:
    """Whether the file exists and its group or others can read or write it."""

    try:
        mode = credentials_path().stat().st_mode
    except FileNotFoundError:
        return False
    return bool(stat.S_IMODE(mode) & 0o077)


def _read() -> dict[str, Any]:
    path = credentials_path()
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ResearchError(
            f"{path} cannot be read as JSON ({type(exc).__name__}). Fix or remove "
            "it, then set the key again with: research-rag api-key set"
        ) from None
    _validate(path, document)
    return {
        name: dict(value)
        for name, value in document.items()
        if name in _PROVIDER_FIELDS
    }


def _validate(path: Path, document: Any) -> None:
    def refuse(field: str, problem: str) -> ResearchError:
        return ResearchError(f"{path}: {field} {problem}. Fix or remove the file.")

    if not isinstance(document, dict):
        raise refuse("the document", "must be an object")
    if document.get("schema_version") != CREDENTIALS_SCHEMA_VERSION:
        raise refuse(
            "schema_version",
            f"must be {CREDENTIALS_SCHEMA_VERSION}; this file is for another version",
        )
    for name, value in document.items():
        if name == "schema_version":
            continue
        if name not in _PROVIDER_FIELDS:
            raise refuse(name, "is not a provider this app knows")
        if not isinstance(value, dict):
            raise refuse(name, "must be an object")
        for field, entry in value.items():
            if field not in _PROVIDER_FIELDS[name]:
                raise refuse(f"{name}.{field}", "is not a field this app knows")
            if not isinstance(entry, str):
                raise refuse(f"{name}.{field}", "must be a string")


def _write(providers: dict[str, Any]) -> None:
    document = {"schema_version": CREDENTIALS_SCHEMA_VERSION, **providers}
    with atomic_file(credentials_path(), private=True) as handle:
        handle.write(json.dumps(document, indent=2, sort_keys=True))
        handle.write("\n")
