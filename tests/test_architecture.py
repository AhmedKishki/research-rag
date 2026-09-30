"""The import boundary: the domain core must not reach for the web layer.

The package is one distribution with two layers. The domain core — the service,
the generation store, extraction, the dense stack, the settings — is reusable on
its own: the command line and the evaluation harness import it directly and call
it in process, while the browser workspace reaches the same state through
`ui.py`. Nothing but this test keeps that arrow pointing one way, so it asserts
both halves of it: only the workspace layer may import the web stack, and no core
module may import that layer.

This app was seeded from an MCP server whose equivalent test read "no core module
may import the MCP surface". The rule is the same one with a different boundary,
and keeping the file rather than deleting it is what stops the arrow from quietly
reversing.

Imports are read with ``ast`` rather than executed, so the test cannot be fooled
by an import that only succeeds in one environment.
"""

from __future__ import annotations

import ast
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent.parent / "src" / "research_rag"

# The modules allowed to import the web stack: the workspace adapter, which
# builds the Starlette application and hosts it, and the typed boundary over the
# vanilla runtime, which reaches that runtime over MCP. Nothing else may. A core
# module that pulls in Starlette or uvicorn would make every command import a web
# framework to do a filesystem read.
WEB_MODULES = frozenset({"ui.py", "ultrarag.py"})

# The modules no core module may import: the workspace layer and the command-line
# surface. `launcher` is deliberately absent — the core reads launcher state for
# `status`, and it is shared plumbing rather than the workspace.
SURFACE_MODULES = frozenset({"cli", "ui"})

# `python -m research_rag` exists to run the command line, so this module is an
# entry point rather than core and is expected to import the surface.
ENTRY_MODULES = frozenset({"__main__.py"})

# The third-party names that make a module part of the web layer. `fastmcp` and
# `mcp` stay listed because the vanilla gateway is reached over MCP, so a core
# module importing either has changed the architecture the same way.
WEB_IMPORTS = frozenset({"fastmcp", "mcp", "starlette", "uvicorn", "ui_ultra_rag_mcp"})


def _modules() -> list[Path]:
    return sorted(path for path in PACKAGE.glob("*.py") if path.name != "__init__.py")


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _absolute_imports(path: Path) -> set[str]:
    """Return every absolute top-level module name this file imports."""

    names: set[str] = set()
    for node in ast.walk(_parse(path)):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    return names


def _package_imports(path: Path) -> set[str]:
    """Return the sibling modules this file imports, `from .x import y` alike."""

    names: set[str] = set()
    for node in ast.walk(_parse(path)):
        if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module:
            names.add(node.module.split(".")[0])
    return names


def test_only_the_workspace_imports_the_web_stack() -> None:
    """A core module that reaches for the web stack has changed the architecture."""

    importers = {
        path.name for path in _modules() if _absolute_imports(path) & WEB_IMPORTS
    }
    assert importers == set(WEB_MODULES), (
        "modules importing the web stack changed; the surface set is "
        f"{sorted(WEB_MODULES)}"
    )


def test_no_core_module_imports_the_surface() -> None:
    """The arrow points one way: the workspace and the entry point use the core."""

    offenders = {
        path.name: sorted(_package_imports(path) & SURFACE_MODULES)
        for path in _modules()
        if path.name not in WEB_MODULES
        and path.name not in ENTRY_MODULES
        # A module that is itself part of the surface is not a core module, so it
        # is exempt in its own right rather than by hosting a web app.
        and path.name.removesuffix(".py") not in SURFACE_MODULES
        and _package_imports(path) & SURFACE_MODULES
    }
    assert offenders == {}, f"core modules importing the surface: {offenders}"


def test_the_mcp_tool_surface_is_gone() -> None:
    """The app answers in process, so no module may declare tools or resources.

    A surviving `@app.tool` or `create_server` would be a second front door that
    nothing tests, and the app has no client to configure it.
    """

    declared: dict[str, str] = {}
    for path in _modules():
        for node in ast.walk(_parse(path)):
            for decorator in getattr(node, "decorator_list", []):
                target = (
                    decorator.func if isinstance(decorator, ast.Call) else decorator
                )
                parts: list[str] = []
                while isinstance(target, ast.Attribute):
                    parts.append(target.attr)
                    target = target.value
                if isinstance(target, ast.Name):
                    parts.append(target.id)
                parts.reverse()
                if parts in (["app", "tool"], ["app", "resource"]):
                    declared[path.name] = f"{'.'.join(parts)}:{node.name}"
    assert declared == {}, f"MCP surface declarations that should not exist: {declared}"


# The layer stack, the registry's `Setting` type, the coercion, the provenance,
# and the three path helpers live in the pinned `config-ultra-rag-mcp` library.
# This app keeps its keys, its packaged default, and its effective settings.
LAYER_MACHINERY = frozenset(
    {
        "LAYER_DEFAULT",
        "Setting",
        "SettingsError",
        "SettingsSources",
        "default_config_path",
        "describe_settings",
        "environment_settings",
        "merge_settings",
        "override_settings",
        "project_config_path",
        "read_config_document",
        "resolve_settings",
        "user_config_path",
    }
)


def test_the_settings_module_defines_no_layer_machinery() -> None:
    """A second copy of the stack is a second set of bugs, and it has happened once."""

    defined = {
        node.name
        for node in _parse(PACKAGE / "settings.py").body
        if isinstance(node, ast.FunctionDef | ast.ClassDef)
    }

    assert not defined & LAYER_MACHINERY, (
        "settings.py defines layer machinery that belongs to the pinned library: "
        f"{sorted(defined & LAYER_MACHINERY)}"
    )
