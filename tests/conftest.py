from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pymupdf
import pytest
from ebooklib import epub
from platformdirs import user_cache_dir

from research_rag.core.records import load_metadata_overrides, write_metadata_overrides
from research_rag.corpus.docling_env import DoclingEnv
from research_rag.project.config import ResearchConfig

# The managed Docling environment a synthetic test stands in for. No interpreter,
# no package, no model, and no network is ever touched: `resolve_docling_env`
# and `ensure_docling_env` are replaced by this value, and the conversion worker
# by `_synthetic_docling_batch`.
_FAKE_DOCLING_ENV = DoclingEnv(
    root=Path("/nonexistent/research-rag-docling-test-env"),
    spec_fingerprint="test-spec",
    versions={"docling": "2.135.0", "rapidocr": "3.9.1"},
)


def _synthetic_docling_batch(
    *,
    source_path: Path,
    start: int,
    end: int,
    offline: bool,
    model_cache_root: Path | None,
    docling_env: DoclingEnv,
    timeout: float = 180,
    worker: Any = None,
) -> dict[str, Any]:
    """A Docling-shaped worker result built from a synthetic test PDF.

    This is test-only. It reads the requested 1-based physical pages with
    pymupdf and emits one prose unit per paragraph, so the staged build, the
    shared cleaning gate, chunk identity, and search behave as they do over a
    real conversion without importing Docling or fetching a model. The payload
    shape is the one ``corpus/extraction.docling_pdf_batch`` consumes.
    """

    del offline, model_cache_root, docling_env, timeout, worker
    pages: dict[str, list[dict[str, Any]]] = {}
    with pymupdf.open(source_path) as document:
        page_count = document.page_count
        metadata = document.metadata or {}
        for page_number in range(max(1, start), min(end, page_count) + 1):
            raw = document[page_number - 1].get_text("text")
            # Paragraph breaks survive; a wrapped line inside a paragraph joins.
            for paragraph in re.split(r"\n\s*\n", raw):
                text = re.sub(r"\s*\n\s*", " ", paragraph).strip()
                if not text:
                    continue
                pages.setdefault(str(page_number), []).append(
                    {
                        "contents": text,
                        "content_kind": "prose",
                        "provenance": [
                            {
                                "page_no": page_number,
                                "charspan": [0, len(text)],
                                "bbox": {"l": 0.0, "t": 0.0, "r": 1.0, "b": 1.0},
                            }
                        ],
                    }
                )
    return {
        "version": "2.135.0",
        "options": {
            "ocr": False,
            "vlm": False,
            "table_structure": False,
            "enrichments": False,
        },
        "boundary": {"memory.max": str(2 * 1024**3)},
        "pages": pages,
        "diagnostics": {
            "retained_item_count": sum(len(units) for units in pages.values()),
            "excluded_furniture_item_count": 0,
            "excluded_picture_child_item_count": 0,
        },
        "pdf_metadata": {
            "title": str(metadata.get("title") or ""),
            "author": str(metadata.get("author") or ""),
        },
        "page_labels": {str(page): str(page) for page in range(1, page_count + 1)},
        "image_only_pages": 0,
    }


@pytest.fixture(autouse=True)
def _synthetic_docling_worker(
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> None:
    """Keep the fast suite offline: never provision or import Docling.

    Docling is the default PDF backend, so an unchanged `write_pdf` test would
    otherwise try to build a managed environment and download a model. The
    ingestion workflow resolves that environment lazily, only when a PDF actually
    extracts, and this fixture replaces it and the conversion worker. A test that
    exercises the real boundary marks itself ``real_docling`` and keeps the real
    code path.

    Tests of the environment resolver keep seeing the real function. A direct
    extraction test requests ``fake_docling_env`` and passes it to
    ``extract_sources`` explicitly.
    """

    if request.node.get_closest_marker("real_docling") is not None:
        return
    # The resolver's own tests must keep exercising the real resolver, including
    # its install path, so they are left untouched.
    module = getattr(request, "module", None)
    if module is not None and module.__name__.endswith("test_docling_env"):
        return
    import research_rag.corpus.docling_env as docling_env_module
    import research_rag.corpus.extraction as extraction_module
    import research_rag.corpus.pdf_backend as pdf_backend_module
    import research_rag.generations.ingestion as ingestion_module

    # Direct extraction helpers call the module-level resolver and pass the
    # result to `extract_sources` explicitly; production still never provisions
    # one on its own.
    monkeypatch.setattr(
        docling_env_module,
        "resolve_docling_env",
        lambda *args, **kwargs: _FAKE_DOCLING_ENV,
    )
    # Backend diagnostics imported the resolver directly. Use the same runtime
    # identity as extraction, rather than the developer's installed environment.
    monkeypatch.setattr(
        pdf_backend_module,
        "resolve_docling_env",
        lambda *args, **kwargs: _FAKE_DOCLING_ENV,
    )
    monkeypatch.setattr(
        ingestion_module,
        "ensure_docling_env",
        lambda *args, **kwargs: _FAKE_DOCLING_ENV,
    )
    monkeypatch.setattr(
        extraction_module,
        "run_docling_batch",
        _synthetic_docling_batch,
    )


@pytest.fixture
def fake_docling_env() -> DoclingEnv:
    """The resolved environment a direct extraction test passes explicitly.

    `extract_sources` keeps taking an explicit environment; this only saves a
    caller from reaching into the autouse stub. It is intentionally not a
    production auto-resolve.
    """

    return _FAKE_DOCLING_ENV


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
    """This suite's embedder cannot tell a repeat from a theme.

    A corpus of two related sentences is one a cosine would collapse, so a number no
    cosine can reach says nothing is close enough, and the tests about the rule pass
    their own threshold, which is the stronger statement.
    """

    monkeypatch.setenv("RESEARCH_RAG_RETRIEVAL_DUPLICATE_COSINE", "2.0")


@pytest.fixture(autouse=True)
def _an_account_directory_of_this_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """Point every account-scoped file at a throwaway directory.

    The project record, the per-user settings file, the command, its menu entries, and its
    icon live in the account's own directories, so a test that runs `install` or reads a
    desktop entry would write there unless it says otherwise.

    `XDG_CACHE_HOME` stays the real cache: it holds only immutable model binaries, and the
    integration test needs the ones already there rather than downloading them again.
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
