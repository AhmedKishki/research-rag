"""Lightweight snapshot paths and required files shared by loading and health.

These checks read local files only. A cached directory alone does not establish
that its ONNX weights and tokenizer are available.
"""

from __future__ import annotations

from pathlib import Path

TOKENIZER_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
)


def snapshot_path(cache_root: Path, repository: str, revision: str) -> Path:
    return (
        cache_root / f"models--{repository.replace('/', '--')}" / "snapshots" / revision
    )


def snapshot_is_complete(snapshot: Path, required_files: tuple[str, ...]) -> bool:
    return all((snapshot / filename).is_file() for filename in required_files)
