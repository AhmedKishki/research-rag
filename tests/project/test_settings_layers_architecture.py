"""What the settings layer is allowed to depend on.

`settings_layers` is configuration layering and nothing else. It acquires a
domain type, a storage layer, or a retrieval import one release at a time
otherwise, so the imports are asserted rather than reviewed: a change that needs
a forbidden import is a change that does not belong here.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

PACKAGE = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "research_rag"
    / "project"
    / "settings_layers"
)

FORBIDDEN = frozenset(
    {
        "fastmcp",
        "mcp",
        "ultrarag",
        "bm25s",
        "fastembed",
        "numpy",
        "onnxruntime",
        "qdrant_client",
        "sqlite3",
        "starlette",
        "uvicorn",
    }
)

ALLOWED_THIRD_PARTY = frozenset({"platformdirs"})


def _imported_roots() -> set[str]:
    """Return the top-level module names every file in the package imports."""

    roots: set[str] = set()
    for source in sorted(PACKAGE.glob("*.py")):
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                roots.add(node.module.split(".")[0])
    return roots


def test_the_package_imports_nothing_forbidden() -> None:
    assert not _imported_roots() & FORBIDDEN


def test_the_third_party_imports_are_platformdirs_only() -> None:
    """`tomllib` is standard library, so one third-party import is the whole list."""

    standard = set(sys.stdlib_module_names) | {"research_rag"}

    third_party = {name for name in _imported_roots() if name not in standard}

    assert third_party == ALLOWED_THIRD_PARTY


def test_the_package_declares_no_registry_of_its_own() -> None:
    """A default file and a key table are the app's, and stay in `settings.py`."""

    from research_rag.project.settings_layers import settings as settings_module

    for name in dir(settings_module):
        if name.startswith("SETTINGS") or name.endswith("_ENVIRONMENT_PREFIX"):
            raise AssertionError(f"the package owns {name}, which belongs to the app")


def test_the_package_ships_no_default_configuration_file() -> None:
    assert list(PACKAGE.glob("*.toml")) == []
