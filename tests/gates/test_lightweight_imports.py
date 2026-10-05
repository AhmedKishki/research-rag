"""A command that only reports must answer on a machine with no model stack.

`AGENTS.md` holds the rule: the app resolves its settings and starts serving
before it opens the gateway, so a project reads fine where the managed runtime is
not installed, and `research-rag config`, `research-rag doctor`, and `install` are
the commands a reader reaches for when something is wrong. That rule is only worth
anything if these commands import without the retrieval stack, so this module makes
each of them run in a subprocess where importing that stack fails outright.

The child blocks the import rather than uninstalling it, so the assertion is about
what a command *reaches for*, not about what happens to be installed here. The
account directories are throwaway, so a command that writes one writes to a
temporary home and never to the account this machine uses.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "src"

#: The stack a lightweight command must not need. `numpy` is here because the
#: dense backends annotate with it, `pymupdf` because a corpus reader pulls it in,
#: and the other three because loading a model needs them.
MODEL_STACK = ("fastembed", "qdrant_client", "pymupdf", "tokenizers", "numpy")

_CHILD = '''
import sys

MODEL_STACK = {stack!r}


class _Blocker:
    """Refuse the model stack the way a machine without it would."""

    def find_module(self, name, path=None):
        return self.find_spec(name, path)

    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in MODEL_STACK:
            raise ImportError(f"{{name}} is not installed on this machine")
        return None


sys.meta_path.insert(0, _Blocker())
sys.argv = {argv!r}

from research_rag.surfaces.cli import main

try:
    main()
except SystemExit as exc:
    status = exc.code
else:
    status = 0

leaked = sorted(name for name in MODEL_STACK if name in sys.modules)
sys.stderr.write("REACHED=" + ",".join(leaked) + "\\n")
sys.stderr.write("STATUS=" + repr(status) + "\\n")
sys.stdout.flush()
'''


def run_without_the_model_stack(
    argv: list[str], home: Path
) -> subprocess.CompletedProcess[str]:
    """Run one command in a subprocess where the model stack cannot be imported."""

    (home / "config" / "research-rag").mkdir(parents=True, exist_ok=True)
    (home / "project" / "sources").mkdir(parents=True, exist_ok=True)
    return subprocess.run(
        [sys.executable, "-c", _CHILD.format(stack=list(MODEL_STACK), argv=argv)],
        check=False,
        capture_output=True,
        text=True,
        timeout=600,
        cwd=str(home / "project"),
        env={
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": str(home),
            "XDG_CONFIG_HOME": str(home / "config"),
            "XDG_DATA_HOME": str(home / "data"),
            "XDG_CACHE_HOME": str(home / "cache"),
            "PYTHONPATH": str(PACKAGE_ROOT),
            "RESEARCH_RAG_OFFLINE": "false",
        },
    )


def _refused_a_missing_stack(completed: subprocess.CompletedProcess[str]) -> str:
    """The reason the child failed, or an empty string when it did not."""

    if "is not installed on this machine" in completed.stderr:
        return "the model stack"
    return ""


@pytest.mark.parametrize(
    ("name", "arguments"),
    (
        ("--version", ["--version"]),
        ("help", ["help"]),
        ("config", ["--project-root", ".", "config"]),
        ("install", ["install", "--help"]),
        ("update", ["update"]),
    ),
)
def test_a_command_that_only_reports_never_reaches_for_the_model_stack(
    tmp_path: Path, name: str, arguments: list[str]
) -> None:
    """Each command runs to completion with the model stack unavailable.

    `update` is here deliberately: it is the one command `AGENTS.md` names as the
    exception, and the reason it was an exception is a module it imported for one
    exception class. That reason no longer holds, so it is held to the same rule.
    """

    completed = run_without_the_model_stack(arguments, tmp_path / "home")

    assert not _refused_a_missing_stack(completed), completed.stderr
    assert "REACHED=\n" in completed.stderr, completed.stderr
    assert (completed.stdout + completed.stderr).count("ImportError") == 0, (
        completed.stderr
    )
    assert "Traceback" not in completed.stderr, completed.stderr


def test_default_doctor_reports_the_missing_stack_rather_than_crashing(
    tmp_path: Path,
) -> None:
    """`doctor` is the command a reader reaches for when something is wrong.

    On a machine without the retrieval stack it cannot read the corpus, so the
    report says which checks that leaves unknown and names the package. It must not
    report an empty project, and it must not fail with a traceback.
    """

    completed = run_without_the_model_stack(
        ["research-rag", "--project-root", ".", "doctor"], tmp_path / "home"
    )

    assert not _refused_a_missing_stack(completed), completed.stderr
    assert "Traceback" not in completed.stderr, completed.stderr
    assert "REACHED=\n" in completed.stderr, completed.stderr
    assert "STATUS=1" in completed.stderr, completed.stderr
    assert "blocked  service" in completed.stdout, completed.stdout
    assert "cannot be read" in completed.stdout, completed.stdout
    # The dependency is named, and the checks that did run are still reported.
    assert "numpy" in completed.stdout or "not importable" in completed.stdout
    assert "blocked  code_currency" not in completed.stdout
    assert "code_currency" in completed.stdout
    assert "runtime_root" in completed.stdout


def test_the_blocked_child_would_have_failed_on_a_reachable_import(
    tmp_path: Path,
) -> None:
    """The blocker works: a command that reaches the stack is stopped by it.

    Without this, a green run of the test above would only prove that the child
    finished, not that anything could have stopped it.
    """

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            _CHILD.format(
                stack=list(MODEL_STACK),
                argv=["research-rag", "--project-root", ".", "search", "heron"],
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=600,
        cwd=str(tmp_path),
        env={
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": str(tmp_path),
            "XDG_CONFIG_HOME": str(tmp_path / "config"),
            "PYTHONPATH": str(PACKAGE_ROOT),
            "RESEARCH_RAG_OFFLINE": "false",
        },
    )

    assert _refused_a_missing_stack(completed), completed.stderr
