from __future__ import annotations

from pathlib import Path
from typing import Any

import pymupdf
import pytest
from ebooklib import epub
from platformdirs import user_cache_dir

from research_rag.config import ResearchConfig
from research_rag.storage import (
    load_metadata_overrides,
    write_metadata_overrides,
)


def write_reviewed_metadata(
    config: ResearchConfig,
    relative_path: str,
    metadata: dict[str, Any],
) -> None:
    """Write one reviewed-metadata override the way a person edits the file."""

    overrides = load_metadata_overrides(config.metadata_path)
    if metadata:
        overrides[relative_path] = metadata
    else:
        overrides.pop(relative_path, None)
    write_metadata_overrides(config.metadata_path, overrides)


def write_pdf(path: Path, pages: list[str], *, title: str = "Test PDF") -> None:
    document = pymupdf.open()
    document.set_metadata({"title": title, "author": "Test Author"})
    for text in pages:
        page = document.new_page()
        page.insert_textbox(
            pymupdf.Rect(72, 72, 540, 760),
            text,
            fontsize=11,
        )
    document.save(path)
    document.close()


def write_epub(path: Path, text: str, *, title: str = "Test EPUB") -> None:
    book = epub.EpubBook()
    book.set_identifier("test-identifier")
    book.set_title(title)
    book.set_language("en")
    book.add_author("EPUB Author")
    chapter = epub.EpubHtml(
        title="Opening Chapter",
        file_name="chapter-1.xhtml",
        lang="en",
    )
    chapter.content = f"<h1>Opening Chapter</h1><p>{text}</p>"
    book.add_item(chapter)
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.toc = (epub.Link("chapter-1.xhtml", "Opening Chapter", "chapter-1"),)
    book.spine = ["nav", chapter]
    epub.write_epub(str(path), book)


@pytest.fixture(autouse=True)
def _no_collapsed_repetitions_in_the_suite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Declare that this suite's embedder cannot tell a repeat from a theme.

    Every build and search here runs on a hand-written embedder that cannot tell
    a repeated passage from two passages about one subject: a corpus of two
    related sentences is one a cosine would collapse. A number no cosine can
    reach says nothing is close enough, and the tests about the rule pass their
    own threshold, which is the stronger statement.
    """

    monkeypatch.setenv("RESEARCH_ULTRARAG_RETRIEVAL_DUPLICATE_COSINE", "2.0")


@pytest.fixture(autouse=True)
def _an_account_directory_of_this_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """Point every account-scoped file at a throwaway directory.

    The project record and the per-user settings file live in the account's
    config directory, and the command, its menu entries, and its icon live in the
    account's data and binary directories, so a test that runs `install` or reads
    a desktop entry writes there unless it says otherwise. Those are the
    reader's own files, and a test run must not add to them or read them.

    `XDG_CACHE_HOME` is pinned to the real cache instead of being redirected: it
    holds only immutable model binaries, and the integration test needs the ones
    that are already there rather than downloading them again.
    """

    account = tmp_path_factory.mktemp("account")
    real_cache = Path(user_cache_dir())
    monkeypatch.setenv("HOME", str(account))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(account / "config"))
    monkeypatch.setenv("XDG_DATA_HOME", str(account / "data"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(real_cache))


@pytest.fixture
def project(tmp_path: Path) -> Path:
    root = tmp_path / "research-project"
    (root / "sources").mkdir(parents=True)
    return root
