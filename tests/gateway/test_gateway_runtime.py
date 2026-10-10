from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from research_rag.gateway import runtime
from research_rag.gateway.manifest import (
    BASELINE_COMMIT,
    BASELINE_VERSION,
    SERVER_SPECS,
)
from research_rag.gateway.runtime import (
    MARKER_FILENAME,
    READ_ONLY_DIRECTORY_MODE,
    READ_ONLY_FILE_MODE,
    RuntimeValidationError,
    _tree_files,
    _tree_hash,
    describe_tree_difference,
    install_managed_runtime,
    make_tree_read_only,
    validate_managed_runtime,
)


def test_gateway_without_optional_library_refuses_before_runtime_download() -> None:
    from research_rag.project.config import child_process_environment

    # Keep the gateway's process-global logging/CLI setup out of the app tests.
    script = """
import importlib.util
import sys
from research_rag.gateway import server
find_spec = importlib.util.find_spec
importlib.util.find_spec = lambda name, *args, **kwargs: None if name == "ultrarag" else find_spec(name, *args, **kwargs)
sys.argv = ["research-rag-gateway", "--workspace-root", "/tmp/kilo"]
def refuse(**kwargs):
    raise AssertionError("missing optional library caused a runtime download")
server.install_managed_runtime = refuse
server.main()
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=child_process_environment(),
        timeout=30,
        check=False,
    )
    assert result.returncode == 2
    message = result.stderr
    assert "uv sync --locked --extra legacy" in message
    assert "Normal research-rag retrieval does not need this gateway" in message


def _make_writable(root: Path) -> None:
    if not root.exists():
        return
    for path in root.rglob("*"):
        os.chmod(path, 0o700 if path.is_dir() else 0o600)
    os.chmod(root, 0o700)


def _write_synthetic_tree(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "pyproject.toml").write_text(
        f'[project]\nname = "UltraRAG"\nversion = "{BASELINE_VERSION}"\n',
        encoding="utf-8",
    )
    for spec in SERVER_SPECS:
        entrypoint = root / spec.relative_entrypoint
        entrypoint.parent.mkdir(parents=True, exist_ok=True)
        entrypoint.write_text(f"# {spec.namespace}\n", encoding="utf-8")


def _pin_tree(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Make this module's pinned constants describe a stand-in tree at root."""
    monkeypatch.setattr(runtime, "TREE_FILES", _tree_files(root))
    monkeypatch.setattr(runtime, "TREE_SHA256", _tree_hash(root))
    (root / MARKER_FILENAME).write_text(
        json.dumps(runtime._expected_marker(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _stub_download(monkeypatch: pytest.MonkeyPatch, extracted: Path) -> None:
    monkeypatch.setattr(runtime, "_download_archive", lambda destination: None)
    monkeypatch.setattr(
        runtime,
        "_safe_extract",
        lambda archive_path, destination: extracted,
    )


@pytest.fixture
def synthetic_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[Path]:
    """An installed managed runtime whose contents this module stands in for."""
    staging = tmp_path / "extracted"
    staging.mkdir()
    extracted = staging / f"UltraRAG-{BASELINE_COMMIT}"
    _write_synthetic_tree(extracted)
    _pin_tree(extracted, monkeypatch)
    _stub_download(monkeypatch, extracted)
    cache = tmp_path / "cache"
    yield install_managed_runtime(cache_root=cache)
    _make_writable(cache)


def _pollute(root: Path) -> Path:
    stray = (
        root / "servers" / "memory" / "src" / "__pycache__" / "memory.cpython-311.pyc"
    )
    stray.parent.mkdir(parents=True, exist_ok=True)
    stray.write_bytes(b"\x00compiled")
    return stray


def test_polluted_tree_names_the_offending_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "UltraRAG"
    _write_synthetic_tree(root)
    _pin_tree(root, monkeypatch)
    stray = _pollute(root)

    with pytest.raises(RuntimeValidationError) as raised:
        validate_managed_runtime(root)

    message = str(raised.value)
    difference = raised.value.difference
    assert difference is not None
    assert difference.kind == "unexpected"
    assert difference.path == stray.relative_to(root).as_posix()
    assert difference.path in message
    assert "unexpected" in message
    assert "Use a fresh cache location" in message
    assert describe_tree_difference(root) == difference


def test_missing_and_changed_paths_are_named(synthetic_runtime: Path) -> None:
    _make_writable(synthetic_runtime)
    entrypoint = synthetic_runtime / SERVER_SPECS[0].relative_entrypoint
    entrypoint.unlink()

    missing = describe_tree_difference(synthetic_runtime)
    assert missing is not None
    assert missing.kind == "missing"
    assert missing.path == SERVER_SPECS[0].relative_entrypoint.as_posix()
    assert missing.mode is None

    _write_synthetic_tree(synthetic_runtime)
    entrypoint.write_text("# changed\n", encoding="utf-8")

    changed = describe_tree_difference(synthetic_runtime)
    assert changed is not None
    assert changed.kind == "changed"
    assert changed.path == SERVER_SPECS[0].relative_entrypoint.as_posix()


def test_missing_marker_is_reported_without_a_difference(
    synthetic_runtime: Path,
) -> None:
    _make_writable(synthetic_runtime)
    (synthetic_runtime / MARKER_FILENAME).unlink()

    with pytest.raises(RuntimeValidationError) as raised:
        validate_managed_runtime(synthetic_runtime)

    assert raised.value.difference is None
    assert MARKER_FILENAME in str(raised.value)


def test_install_leaves_the_verified_tree_read_only(synthetic_runtime: Path) -> None:
    assert stat.S_IMODE(synthetic_runtime.stat().st_mode) == READ_ONLY_DIRECTORY_MODE
    for path in synthetic_runtime.rglob("*"):
        expected = READ_ONLY_DIRECTORY_MODE if path.is_dir() else READ_ONLY_FILE_MODE
        assert stat.S_IMODE(path.stat().st_mode) == expected, path


def test_offline_validation_accepts_a_read_only_tree(
    synthetic_runtime: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "research-rag-runtime",
            "--offline",
            "--cache-root",
            str(synthetic_runtime.parents[1]),
        ],
    )
    runtime.main()

    assert capsys.readouterr().out.strip() == str(synthetic_runtime)


def test_read_only_tree_still_reports_pollution(synthetic_runtime: Path) -> None:
    _make_writable(synthetic_runtime)
    stray = _pollute(synthetic_runtime)

    with pytest.raises(RuntimeValidationError) as raised:
        validate_managed_runtime(synthetic_runtime)

    assert raised.value.difference is not None
    assert (
        raised.value.difference.path == stray.relative_to(synthetic_runtime).as_posix()
    )


def test_non_posix_platform_reports_the_skipped_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    staging = tmp_path / "extracted"
    staging.mkdir()
    extracted = staging / f"UltraRAG-{BASELINE_COMMIT}"
    _write_synthetic_tree(extracted)
    _pin_tree(extracted, monkeypatch)
    _stub_download(monkeypatch, extracted)
    monkeypatch.setattr(runtime, "_posix_modes_available", lambda: False)
    cache = tmp_path / "cache"

    installed = install_managed_runtime(cache_root=cache)
    captured = capsys.readouterr()

    assert os.access(installed / SERVER_SPECS[0].relative_entrypoint, os.W_OK)
    assert "left writable" in captured.err
    assert "read-only" not in captured.err
    assert validate_managed_runtime(installed) == installed
    _make_writable(cache)


def test_make_tree_read_only_reports_only_when_skipped(
    synthetic_runtime: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _make_writable(synthetic_runtime)
    monkeypatch.setattr(runtime, "_posix_modes_available", lambda: False)
    assert "left writable" in (make_tree_read_only(synthetic_runtime) or "")

    monkeypatch.setattr(runtime, "_posix_modes_available", lambda: True)
    assert make_tree_read_only(synthetic_runtime) is None
    assert stat.S_IMODE(synthetic_runtime.stat().st_mode) == READ_ONLY_DIRECTORY_MODE
