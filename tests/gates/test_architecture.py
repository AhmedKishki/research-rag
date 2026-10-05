"""Nothing but this test keeps the package split at `surfaces/`.

A surface that imported another would be a second copy of an operation. Imports are
read with `ast` rather than executed, so an import that succeeds only in one
environment cannot fool the test.
"""

from __future__ import annotations

import ast
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[2] / "src" / "research_rag"
SURFACES = PACKAGE / "surfaces"

# The modules allowed to import the web stack. `app.py` composes the one ASGI
# application; `control.py` is the control API and the handle the command line
# holds on it; `surfaces/ui.py` builds the workspace and its adapter;
# `surfaces/workspace/app.py` is the workspace that adapter serves;
# `surfaces/mcp.py` declares the agent's tools; and `bridge.py` proxies stdio to
# the app the workspace and the agent surface are already served by.
# `ultrarag.py` is the typed boundary over the gateway, which is reached over
# MCP, so it has always been allowed to name FastMCP, and `gateway/server.py` is
# that gateway: the process this app spawns and speaks MCP to.
#
# `surfaces/cli.py` is deliberately absent: the command line reaches the app
# through `app.py` and `control.py` and imports no web stack of its own, which is
# what makes `research-rag config` and `research-rag doctor` work on a machine
# with no Starlette importable.
WEB_MODULES = frozenset(
    {
        "runtime/app.py",
        "surfaces/bridge.py",
        "runtime/control.py",
        "surfaces/workspace/write_guard.py",
        "gateway/server.py",
        "surfaces/mcp.py",
        "surfaces/ui.py",
        "surfaces/workspace/app.py",
        "retrieval/ultrarag.py",
    }
)

# The modules a surface may not import. The workspace, the agent tools, and the
# command line each name their own arguments, so a shared table would be a
# second place for an operation to be wrong.
CROSS_SURFACE = {
    "surfaces/cli.py": frozenset({"surfaces/mcp.py", "surfaces/ui.py"}),
    "surfaces/mcp.py": frozenset({"surfaces/cli.py", "surfaces/ui.py"}),
    "surfaces/ui.py": frozenset({"surfaces/cli.py", "surfaces/mcp.py"}),
}

# `mcp` stays listed beside `fastmcp` because the gateway is reached over MCP, so an
# engine module importing either has changed the architecture.
WEB_IMPORTS = frozenset({"fastmcp", "mcp", "pydantic", "starlette", "uvicorn"})


def _modules() -> list[Path]:
    return sorted(
        path
        for path in PACKAGE.rglob("*.py")
        if path.name not in {"__init__.py", "__main__.py"}
    )


def _relative(relative: str) -> str:
    return relative.replace("\\", "/")


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _absolute_imports(path: Path) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(_parse(path)):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    return names


def _package_imports(path: Path) -> set[str]:
    """Both spellings of a relative import, reported as package-relative names.

    A module under `surfaces/` names its siblings without the package prefix and its
    engine modules with one.
    """

    names: set[str] = set()
    depth = len(path.relative_to(PACKAGE).parts) - 1
    for node in ast.walk(_parse(path)):
        if not isinstance(node, ast.ImportFrom) or node.level == 0:
            continue
        if not node.module:
            continue
        if node.level == depth + 1 or node.level <= depth:
            names.add(f"{node.module.split('.')[0]}.py")
        else:
            prefix = "/".join(path.relative_to(PACKAGE).parts[: node.level - 1])
            names.add(f"{prefix}/{node.module.split('.')[0]}.py")
    return names


def test_only_a_surface_imports_the_web_stack() -> None:
    importers = {
        _relative(path.relative_to(PACKAGE).as_posix())
        for path in _modules()
        if _absolute_imports(path) & WEB_IMPORTS
    }
    assert importers == set(WEB_MODULES), (
        "modules importing the web stack changed; the surface set is "
        f"{sorted(WEB_MODULES)}"
    )


def test_no_engine_module_imports_a_surface() -> None:
    offenders: dict[str, list[str]] = {}
    for path in _modules():
        if path.is_relative_to(SURFACES):
            continue
        imported = [
            name for name in _package_imports(path) if name.startswith("surfaces/")
        ]
        if imported:
            offenders[_relative(path.relative_to(PACKAGE).as_posix())] = sorted(
                imported
            )
    assert offenders == {}, f"engine modules importing a surface: {offenders}"


def test_no_surface_imports_another_surface() -> None:
    offenders = {
        relative: sorted(_package_imports(path) & CROSS_SURFACE[relative])
        for path in _modules()
        if (relative := _relative(path.relative_to(PACKAGE).as_posix()))
        in CROSS_SURFACE
        and _package_imports(path) & CROSS_SURFACE[relative]
    }
    assert offenders == {}, f"surfaces importing one another: {offenders}"


def test_the_agent_surface_is_declared_once() -> None:
    """One tool, one resource, and `status` twice.

    `status` is declared once for a project this machine serves and once for a project
    its agent entry names and nothing here resolves. Every other operation is declared
    once, because a second declaration would be a second answer for a corpus that
    exists.
    """

    declaring: dict[str, list[str]] = {}
    for path in _modules():
        names: list[str] = []
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
                if parts[:1] == ["app"] and parts[1:2] in (["tool"], ["resource"]):
                    names.append(f"{'.'.join(parts)}:{node.name}")
        if names:
            declaring[_relative(path.relative_to(PACKAGE).as_posix())] = names
    assert list(declaring) == ["surfaces/mcp.py"], declaring

    operations = [name.split(":", 1)[1] for name in declaring["surfaces/mcp.py"]]
    assert set(operations) == {
        "status",
        "ingest",
        "search",
        "find_source",
        "get_passage",
        "set_source_inclusion",
        "set_chunk_inclusion",
        "set_source_metadata",
        "status_resource",
    }
    assert len(operations) == 11, operations
    for name in set(operations) - {"status", "status_resource"}:
        assert operations.count(name) == 1, name
    assert operations.count("status") == 2


def test_the_answer_projection_is_shared_by_the_two_bounded_readers() -> None:
    """An agent and a terminal read one projection; the workspace reads the payload.

    A second reader of the lean answer must read this projection rather than write its
    own, and the workspace, with the whole screen, must not reach for it.
    """

    callers = sorted(
        _relative(path.relative_to(PACKAGE).as_posix())
        for path in _modules()
        if "present_tool_response" in path.read_text(encoding="utf-8")
        and path.name != "tool_views.py"  # the module that defines it
    )
    assert callers == ["surfaces/mcp.py"], callers
    # The command line selects the mode per run rather than through the MCP server's own
    # configuration, and the account record reads the projection because reading
    # whether one project's app is up is part of reading the record.
    lean_callers = sorted(
        _relative(path.relative_to(PACKAGE).as_posix())
        for path in _modules()
        if "lean_status" in path.read_text(encoding="utf-8")
        and path.name != "tool_views.py"
    )
    assert lean_callers == ["project/registry.py", "surfaces/cli.py"], lean_callers
    # The workspace, with the whole payload on screen, still must not reach for
    # it: it reads the service's own answer rather than a projection of it.
    assert "surfaces/ui.py" not in lean_callers


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
