"""Nothing but this test keeps the package's folders in order.

The folders are layered, and a module imports from its own folder and the folders
below it, so no folder can grow a dependency on one that builds on it. A surface
that imported another would be a second copy of an operation. Imports are read
with `ast` rather than executed, so an import that succeeds only in one
environment cannot fool the test.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator
from functools import cache
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[2] / "src" / "research_rag"

# The folders, lowest first. `gateway` holds the retired command shims, which
# import nothing.
LAYERS = (
    "storage",
    "project",
    "corpus",
    "retrieval",
    "generations",
    "synthesis",
    "core",
    "runtime",
    "surfaces",
    "gateway",
)

# The imports that cross the order upward. `runtime/app.py` composes the one
# application, so it mounts the surfaces it serves; `runtime/control.py` holds the
# control API to the workspace's write guard, which imports nothing outside the
# workspace, and reaching it loads the workspace package.
UPWARD = {
    "runtime/app.py": frozenset({"surfaces/mcp.py", "surfaces/ui.py"}),
    "runtime/control.py": frozenset(
        {"surfaces/workspace/__init__.py", "surfaces/workspace/write_guard.py"}
    ),
}

# The surfaces, each a set of modules that may import only one another: the
# command line, the agent surface and the stdio bridge that proxies it, and the
# workspace with this app's adapter over it. The command line's `mcp` command runs
# the bridge.
SURFACES = {
    "cli": ("surfaces/cli/",),
    "agent": ("surfaces/mcp.py", "surfaces/bridge.py"),
    "workspace": ("surfaces/ui.py", "surfaces/workspace/"),
}
CROSS_SURFACE = {"surfaces/cli/__init__.py": frozenset({"surfaces/bridge.py"})}

# The modules allowed to import the web stack. `app.py` composes the one ASGI
# application; `control.py` is the control API and the handle the command line
# holds on it; `surfaces/ui.py` builds the workspace and its adapter;
# `surfaces/workspace/app.py` is the workspace that adapter serves;
# `surfaces/mcp.py` declares the agent's tools; and `bridge.py` proxies stdio to
# the app the workspace and the agent surface are already served by.
# Retired gateway command shims import no web stack or external framework.
#
# `surfaces/cli/` is deliberately absent: the command line reaches the app
# through `app.py` and `control.py` and imports no web stack of its own, which is
# what makes `research-rag config` and `research-rag doctor` work on a machine
# with no Starlette importable.
WEB_MODULES = frozenset(
    {
        "runtime/app.py",
        "surfaces/bridge.py",
        "runtime/control.py",
        "surfaces/workspace/write_guard.py",
        "surfaces/mcp.py",
        "surfaces/ui.py",
        "surfaces/workspace/app.py",
    }
)

# An engine module importing either MCP implementation crosses the surface boundary.
WEB_IMPORTS = frozenset({"fastmcp", "mcp", "pydantic", "starlette", "uvicorn"})


def _modules() -> list[Path]:
    return sorted(
        path
        for path in PACKAGE.rglob("*.py")
        if path.name not in {"__init__.py", "__main__.py"}
    )


def _relative(path: Path) -> str:
    return path.relative_to(PACKAGE).as_posix()


@cache
def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


@pytest.fixture(scope="module", autouse=True)
def _parsed_source_snapshot() -> Iterator[None]:
    """Share trees within a gate run, not across subsequent pytest invocations."""
    _parse.cache_clear()
    yield
    _parse.cache_clear()


def _absolute_imports(path: Path) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(_parse(path)):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    return names


def _module_file(directory: Path, parts: list[str]) -> Path | None:
    """The file a dotted name under `directory` names: a module or a package."""

    target = directory.joinpath(*parts)
    if target.with_suffix(".py").is_file():
        return target.with_suffix(".py")
    if (target / "__init__.py").is_file():
        return target / "__init__.py"
    return None


def _package_imports(path: Path) -> set[str]:
    """The package modules one module imports, as paths relative to the package.

    A relative import is resolved from the module's own directory, `from . import
    name` names a module, and an absolute import of `research_rag` is resolved from
    the package root, so every spelling of one import resolves to one file.
    """

    found: set[str] = set()
    for node in ast.walk(_parse(path)):
        if isinstance(node, ast.Import):
            modules = [
                alias.name.split(".")[1:]
                for alias in node.names
                if alias.name.split(".")[0] == PACKAGE.name
            ]
            targets = [_module_file(PACKAGE, parts) for parts in modules]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = path.parent
                for _ in range(node.level - 1):
                    base = base.parent
                parts = (node.module or "").split(".") if node.module else []
            elif (node.module or "").split(".")[0] == PACKAGE.name:
                base, parts = PACKAGE, node.module.split(".")[1:]
            else:
                continue
            targets = [_module_file(base, parts) if parts else None]
            # `from .package import module` imports the module too.
            targets += [
                _module_file(base, [*parts, alias.name]) for alias in node.names
            ]
        else:
            continue
        found.update(_relative(target) for target in targets if target is not None)
    found.discard(_relative(path))
    return found


def _folder(relative: str) -> str:
    return relative.split("/", 1)[0] if "/" in relative else ""


def _surface(relative: str) -> str | None:
    return next(
        (
            name
            for name, members in SURFACES.items()
            if any(
                relative == member or relative.startswith(member) for member in members
            )
        ),
        None,
    )


def test_the_gate_reads_imports() -> None:
    """A resolver that found nothing would pass every rule below."""

    imports = _package_imports(PACKAGE / "retrieval" / "search.py")
    assert {"project/policy.py", "retrieval/documents.py"} <= imports, imports
    assert "surfaces/bridge.py" in _package_imports(
        PACKAGE / "surfaces" / "cli" / "__init__.py"
    )


def test_every_folder_has_a_layer() -> None:
    folders = {
        path.name
        for path in PACKAGE.iterdir()
        if path.is_dir() and any(path.glob("*.py"))
    }
    assert folders == set(LAYERS)


def test_a_folder_imports_only_itself_and_the_folders_below() -> None:
    offenders: dict[str, list[str]] = {}
    for path in sorted(PACKAGE.rglob("*.py")):
        relative = _relative(path)
        folder = _folder(relative)
        if not folder:
            continue
        upward = sorted(
            imported
            for imported in _package_imports(path)
            if _folder(imported) != folder
            and LAYERS.index(_folder(imported)) > LAYERS.index(folder)
            and imported not in UPWARD.get(relative, ())
        )
        if upward:
            offenders[relative] = upward
    assert offenders == {}, f"modules importing a folder above their own: {offenders}"


def test_only_a_surface_imports_the_web_stack() -> None:
    importers = {
        _relative(path) for path in _modules() if _absolute_imports(path) & WEB_IMPORTS
    }
    offenders = importers - set(WEB_MODULES)
    assert not offenders, (
        f"modules importing the web stack outside its boundary: {sorted(offenders)}"
    )


def test_no_module_imports_the_retired_ultrarag_library() -> None:
    offenders = [
        _relative(path)
        for path in sorted(PACKAGE.rglob("*.py"))
        if "ultrarag" in _absolute_imports(path)
    ]
    assert offenders == [], f"modules still importing UltraRAG: {offenders}"


def test_no_surface_imports_another_surface() -> None:
    offenders: dict[str, list[str]] = {}
    for path in sorted((PACKAGE / "surfaces").rglob("*.py")):
        relative = _relative(path)
        surface = _surface(relative)
        crossing = sorted(
            imported
            for imported in _package_imports(path)
            if (other := _surface(imported)) is not None
            and other != surface
            and imported not in CROSS_SURFACE.get(relative, ())
        )
        if crossing:
            offenders[relative] = crossing
    assert offenders == {}, f"surfaces importing one another: {offenders}"


def test_the_workspace_imports_nothing_outside_itself() -> None:
    """A host that stores state in a shared package cannot say where it is."""

    offenders = {
        _relative(path): sorted(
            imported
            for imported in _package_imports(path)
            if not imported.startswith("surfaces/workspace/")
        )
        for path in sorted((PACKAGE / "surfaces" / "workspace").rglob("*.py"))
    }
    assert {name: found for name, found in offenders.items() if found} == {}


# The layer stack, the registry's `Setting` type, the coercion, the provenance, and the
# three path helpers live in `settings_layers`. This app keeps its keys, its packaged
# default, and its effective settings.
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
    defined = {
        node.name
        for node in _parse(PACKAGE / "project" / "settings.py").body
        if isinstance(node, ast.FunctionDef | ast.ClassDef)
    }

    assert not defined & LAYER_MACHINERY, (
        "settings.py defines layer machinery that belongs to settings_layers: "
        f"{sorted(defined & LAYER_MACHINERY)}"
    )


def test_the_entry_point_runs_the_command_line() -> None:
    """`python -m research_rag` is how a checkout is run before it is installed."""

    assert "surfaces.cli" in (PACKAGE / "__main__.py").read_text(encoding="utf-8")
