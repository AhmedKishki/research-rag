"""The gateway and runtime console commands are retirement shims.

Each keeps its historical name so a client or wrapper that still calls it fails
with migration guidance instead of a missing executable. Neither starts a
process, imports UltraRAG, reads or writes a cache, or creates a path.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

GATEWAY = Path(__file__).resolve().parents[2] / "src" / "research_rag" / "gateway"

# A retirement shim has no reason to reach for the upstream library, an MCP
# stack, or anything that starts a child, opens a socket, or unpacks an archive.
FORBIDDEN_IMPORT_ROOTS = frozenset(
    {
        "fastmcp",
        "mcp",
        "ultrarag",
        "subprocess",
        "urllib",
        "socket",
        "http",
        "tarfile",
        "platformdirs",
        "tomllib",
    }
)

# Runs one shim where the upstream and process stack cannot be imported. A shim
# that reaches for any of them fails loudly here.
_CHILD = """
import sys

# stdlib pathlib needs urllib.parse, so only the network opener is blocked.
BLOCKED_ROOTS = {"fastmcp", "mcp", "ultrarag", "subprocess", "socket", "tarfile"}
BLOCKED_NAMES = {"urllib.request"}


class _Blocker:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in BLOCKED_ROOTS or name in BLOCKED_NAMES:
            raise ImportError(f"{name} must not be imported by a retirement shim")
        return None


sys.meta_path.insert(0, _Blocker())
sys.argv = __ARGV__

from research_rag.gateway import __MODULE__

try:
    __MODULE__.main()
except SystemExit as exc:
    code = exc.code
    if isinstance(code, str) and code:
        sys.stderr.write(code + "\\n")
    sys.stderr.write("EXIT=" + repr(code) + "\\n")
else:
    sys.stderr.write("EXIT=None\\n")
"""


def _import_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            roots.add(node.module)
    return {name.split(".")[0] for name in roots}


def test_the_shims_import_no_upstream_or_process_stack() -> None:
    for name in ("server.py", "runtime.py"):
        roots = _import_roots(GATEWAY / name)
        assert not (roots & FORBIDDEN_IMPORT_ROOTS), (name, roots)


def test_gateway_help_explains_retirement_and_needs_no_workspace_root(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from research_rag.gateway import server

    monkeypatch.setattr(sys, "argv", ["research-rag-gateway", "--help"])
    with pytest.raises(SystemExit) as raised:
        server.main()

    assert raised.value.code == 0
    out = capsys.readouterr().out
    assert "retired" in out.lower()
    assert "research-rag mcp --project-name" in out


def test_gateway_refuses_with_migration_guidance_and_creates_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from research_rag.gateway import server

    workspace = tmp_path / "workspace"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "research-rag-gateway",
            "--namespace",
            "corpus",
            "--namespace",
            "retriever",
            "--ultrarag-root",
            str(tmp_path / "ultrarag"),
            "--runtime-cache-root",
            str(tmp_path / "cache"),
            "--workspace-root",
            str(workspace),
            "--offline",
            "--log-level",
            "debug",
        ],
    )

    with pytest.raises(SystemExit) as raised:
        server.main()

    message = str(raised.value)
    assert "retired" in message.lower()
    assert "research-rag mcp --project-name" in message
    # An invocation with no --workspace-root still reaches the guidance rather
    # than an argparse error, and no synthetic path is created.
    assert not workspace.exists()
    assert not (tmp_path / "ultrarag").exists()
    assert not (tmp_path / "cache").exists()


def test_runtime_reports_no_external_runtime_and_exits_zero(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    from research_rag.gateway import runtime

    cache = tmp_path / "cache"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "research-rag-runtime",
            "--cache-root",
            str(cache),
            "--offline",
            "--validate-only",
        ],
    )
    runtime.main()

    out = capsys.readouterr().out
    assert "No external runtime is required" in out
    assert not cache.exists()


@pytest.mark.parametrize(
    ("module", "argv"),
    (
        (
            "server",
            [
                "research-rag-gateway",
                "--workspace-root",
                "/nonexistent",
                "--offline",
            ],
        ),
        ("runtime", ["research-rag-runtime", "--offline", "--validate-only"]),
    ),
)
def test_the_shims_run_without_the_upstream_or_process_stack(
    tmp_path: Path, module: str, argv: list[str]
) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            _CHILD.replace("__MODULE__", module).replace("__ARGV__", repr(argv)),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
        cwd=str(tmp_path),
    )

    assert "must not be imported" not in completed.stderr, completed.stderr
    assert "Traceback" not in completed.stderr, completed.stderr
    if module == "server":
        assert "EXIT=" in completed.stderr
        assert "research-rag mcp --project-name" in completed.stderr
    else:
        assert "EXIT=None" in completed.stderr
        assert "No external runtime is required" in completed.stdout
