"""OCR is gone: no command, no module, no extra, and no document naming one.

Scanned PDFs need OCR done outside this app. Ingestion reads a text layer and
nothing else, so the removal is guarded here as well as in the documentation gate.
"""

from __future__ import annotations

import importlib
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DOCUMENTS = (
    "README.md",
    "FEATURES.md",
    "ROADMAP.md",
    "AGENTS.md",
    "src/research_rag/corpus/README.md",
    "tests/corpus/README.md",
)


def test_the_parser_no_longer_offers_ocr() -> None:
    from research_rag.surfaces.cli import _parser

    with pytest.raises(SystemExit):
        _parser().parse_args(["ocr", "scan.pdf"])


def test_the_help_menu_no_longer_lists_ocr() -> None:
    from research_rag.surfaces.cli import HELP_GROUPS

    listed = [name for _, entries in HELP_GROUPS for name, _ in entries]
    assert "ocr" not in listed


def test_the_ocr_module_is_gone() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("research_rag.corpus.ocr")


def test_there_is_no_ocr_extra() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    extras = project["project"].get("optional-dependencies", {})
    assert "ocr" not in extras


def test_no_shipped_document_names_the_removed_command() -> None:
    for document in DOCUMENTS:
        text = (ROOT / document).read_text(encoding="utf-8")
        assert "research-rag ocr" not in text, document
