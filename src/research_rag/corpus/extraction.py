"""Source extraction: Docling PDF routing, EPUB/MOBI units, and the shared gate.

The originals are the quote authority, so this module never rewrites a source and
never writes to `sources/`. A PDF is converted by the managed Docling worker in
`pdf_backend`; this module adapts that payload into units and never reads a PDF's
own text. EPUB and MOBI are read in process. It imports PyMuPDF (page-count
probing only), ebooklib, and BeautifulSoup at module level and therefore cannot be
imported where the retrieval stack is absent.

Normalization of extracted text lives in `text_normalization`, and the reasons a
text is withheld live in `text_quality`. Both are pure string modules, so a
caller that only needs to know whether text is usable imports neither of the
document libraries.
"""

from __future__ import annotations

import hashlib
import re
import threading
from collections import Counter
from pathlib import Path
from typing import Any

import ebooklib
import pymupdf
from bs4 import BeautifulSoup, Comment, NavigableString, PageElement, Tag
from ebooklib import epub

from ..project.settings import BM25_STOPWORD_LANGUAGES, bm25_stopwords
from .docling_env import DoclingEnv
from .mobi_reader import MobiReadError, MobiText, extract_mobi_payload
from .pdf_backend import (
    DoclingConversionError,
    DoclingWorker,
    docling_backend_record_from_env,
    run_docling_batch,
)
from .sources import SourceFile, sha256_file
from .text_normalization import (
    clean_unclean_passage,
    ends_sentence,
    normalize_inline_text,
    normalize_reading_text,
)
from .text_quality import (
    EXTRACTION_ARTIFACT_TOKEN,
    SOURCE_REASON_NO_TEXT,
    SOURCE_REASON_NO_TEXT_LAYER,
    SOURCE_REASON_UNCLEAN,
    clean_damaged_pdf_passage,
    has_searchable_alphanumeric_content,
    non_argument_removal_flags,
    passage_health_reasons,
    source_health_reasons,
    text_corruption_reasons,
    text_health_reasons,
)


class ExtractionError(RuntimeError):
    pass


class UncleanSourceError(ExtractionError):
    """The source health gate refused unreadable evidence, not an I/O failure."""


_DOI = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.IGNORECASE)
_URL = re.compile(r"^(?:https?://|www\.|(?:dx\.)?doi\.org/)", re.IGNORECASE)
_TITLE_REJECTIONS = (
    re.compile(r"^untitled$", re.IGNORECASE),
    re.compile(
        r"^(?:executive\s+summary|summary|abstract|foreword|preface|introduction|"
        r"contents|table\s+of\s+contents)$",
        re.IGNORECASE,
    ),
    re.compile(r"^microsoft\s+word\s*[-–:]", re.IGNORECASE),
    re.compile(r"^adobe\s+(?:acrobat|indesign)", re.IGNORECASE),
    re.compile(r"^(?:document|export|file)(?:[-_ ]?\d+)?$", re.IGNORECASE),
)
_FRONT_MATTER_EXCLUSIONS = re.compile(
    r"^(?:abstract|keywords?|key\s+words?|contents|table\s+of\s+contents|"
    r"volume\s+\d+|number\s+\d+|issn\b|doi\b|copyright\b|©|arxiv:|"
    r"check\s+for\s+updates$)",
    re.IGNORECASE,
)
_YEAR = re.compile(r"\b(18\d{2}|19\d{2}|20\d{2}|21\d{2})\b")
_PDF_DATE_YEAR = re.compile(r"^(?:D:)?(18\d{2}|19\d{2}|20\d{2}|21\d{2})")
_BYLINE = re.compile(r"^(?:by|written\s+by|author(?:s)?\s*:)\s+", re.IGNORECASE)
_AUTHOR_JUNK = re.compile(
    r"^(?:unknown|anonymous|untitled|microsoft\s+word|adobe\s+(?:acrobat|indesign)|"
    r"administrator|admin|user)$",
    re.IGNORECASE,
)
_AUTHOR_AFFILIATION = re.compile(
    r"\s*\([^)]*(?:university|institute|college)[^)]*\)\s*$", re.IGNORECASE
)
_AUTHOR_MARKER = re.compile(r"(?:\d+(?:\s*,\s*\d+)*|[*∗†‡§])+$")
_EPUB_HEADING_ELEMENTS = frozenset({"h1", "h2", "h3", "h4", "h5", "h6"})
_EPUB_CONTENT_ELEMENTS = frozenset({"p", "li", "blockquote", "table"})
_EPUB_SEMANTIC_ELEMENTS = _EPUB_HEADING_ELEMENTS | _EPUB_CONTENT_ELEMENTS
# A note reference prints a number or a note symbol, sometimes bracketed, and
# links to its note. The inline elements a publisher wraps it in print nothing
# else, so they go with it.
_EPUB_NOTE_MARKER_TEXT = re.compile(r"[\[(]?(?:\d{1,4}|[*∗†‡§¶]{1,3})[\])]?")
_EPUB_MARKER_WRAPPERS = frozenset({"sup", "span", "small"})


REMOVAL_FIELDS = (
    "removed_repeated_margin_blocks",
    "removed_sidebar_blocks",
    "removed_footnote_blocks",
    "removed_reference_segments",
    "removed_note_segments",
    "removed_non_argument_characters",
    "removed_epub_furniture_elements",
    "image_only_pages",
    "pdf_text_recovery_pages",
    "recovered_pdf_blocks",
)


def empty_removal_counts() -> dict[str, int]:
    """Every removal category at zero, for a record that has removed nothing yet.

    A caller builds this once and adds into it, so the categories present in a
    document record are the same set whatever path produced them.
    """

    return dict.fromkeys(REMOVAL_FIELDS, 0)


def merge_removal_counts(
    accumulated: dict[str, int] | None,
    added: dict[str, int],
) -> dict[str, int]:
    """Add one batch's removal counts into the running totals of a source."""

    totals = empty_removal_counts()
    for source in (accumulated or {}, added):
        for key, value in source.items():
            if key in totals:
                totals[key] += int(value)
    return totals


def _normalize_text_list(values: list[Any]) -> list[str]:
    return [
        normalized
        for value in values
        if (normalized := normalize_inline_text(str(value)))
    ]


def _document_id(source: SourceFile, digest: str) -> str:
    identity = f"{source.source_relative_path}\0{digest}".encode()
    return f"doc_{hashlib.sha256(identity).hexdigest()[:24]}"


def _doi(value: str) -> str:
    match = _DOI.search(value)
    if match is None:
        return ""
    return match.group(0).rstrip(".,;:)]}")


def _valid_title(
    value: str,
    filename_stem: str,
    *,
    reject_filename_match: bool = True,
) -> bool:
    normalized = normalize_inline_text(value)
    if not normalized or len(normalized) > 500:
        return False
    doi_candidate = re.sub(r"^doi\s*:\s*", "", normalized, flags=re.IGNORECASE)
    if _DOI.fullmatch(doi_candidate) or _URL.match(normalized):
        return False
    if any(pattern.search(normalized) for pattern in _TITLE_REJECTIONS):
        return False
    comparable_title = re.sub(
        r"\.(?:pdf|epub|mobi)$", "", normalized, flags=re.IGNORECASE
    )
    comparable = re.sub(r"[^a-z0-9]+", "", comparable_title.casefold())
    filename = re.sub(r"[^a-z0-9]+", "", filename_stem.casefold())
    if reject_filename_match and comparable and comparable == filename:
        return False
    return sum(character.isalpha() for character in normalized) >= 3


def _valid_author(value: str) -> bool:
    normalized = normalize_inline_text(value)
    doi_candidate = re.sub(r"^doi\s*:\s*", "", normalized, flags=re.IGNORECASE)
    return bool(
        normalized
        and len(normalized) <= 240
        and not _URL.match(normalized)
        and not _DOI.fullmatch(doi_candidate)
        and not _AUTHOR_JUNK.fullmatch(normalized)
        and sum(character.isalpha() for character in normalized) >= 2
    )


def _clean_author_name(value: str) -> str:
    normalized = _BYLINE.sub("", normalize_inline_text(value)).strip()
    normalized = re.sub(r"^(?:and|&)\s+", "", normalized, flags=re.IGNORECASE)
    normalized = _AUTHOR_AFFILIATION.sub("", normalized)
    normalized = _AUTHOR_MARKER.sub("", normalized).strip(" ,;|")
    return normalized


def _author_names(value: str) -> list[str]:
    normalized = normalize_inline_text(value)
    explicitly_by = bool(_BYLINE.match(normalized))
    cleaned = _BYLINE.sub("", normalized).strip()
    if (
        not cleaned
        or _YEAR.fullmatch(cleaned)
        or _URL.match(cleaned)
        or ":" in cleaned
        or re.search(
            r"\b(?:journal|papers?|volume|issue|no\.?|issn)\b",
            cleaned,
            re.IGNORECASE,
        )
    ):
        return []
    if ends_sentence(cleaned) or len(cleaned.split()) > 32:
        return []

    has_list_separator = bool(re.search(r"\s(?:&|and)\s|;", cleaned, re.IGNORECASE))
    if cleaned.count(",") >= 2 or has_list_separator:
        parts = re.split(r"\s*(?:,|;|\s&\s|\sand\s)\s*", cleaned, flags=re.IGNORECASE)
    else:
        parts = [cleaned]
    names = [name for part in parts if (name := _clean_author_name(part))]
    if not names or any(not _valid_author(name) for name in names):
        return []

    organization_words = {
        "agency",
        "association",
        "commission",
        "committee",
        "council",
        "department",
        "foundation",
        "institute",
        "international",
        "laboratory",
        "organisation",
        "organization",
        "university",
    }
    if explicitly_by or len(names) > 1:
        return names
    words = names[0].split()
    capitalized = sum(word[:1].isupper() for word in words if word)
    is_organization = any(
        word.casefold().strip(".,") in organization_words for word in words
    )
    if is_organization or (2 <= len(words) <= 8 and capitalized >= 2):
        return names
    return []


# A language is detected from how much of a language's own function-word
# inventory a text sample uses. Coverage, not raw share, because bm25s's lists are
# very different sizes (33 English words against 499 Korean ones) and a share
# would hand every sample to the longest list.
_LANGUAGE_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)
_LANGUAGE_SAMPLE_TOKENS = 4000
_LANGUAGE_MINIMUM_TOKENS = 120
_LANGUAGE_MINIMUM_COVERAGE = 0.10
_LANGUAGE_MINIMUM_MARGIN = 1.5
_LANGUAGE_TAG = re.compile(r"^([A-Za-z]{2,3})(?:[-_].*)?$")


def _iso_language_code(value: str) -> str:
    match = _LANGUAGE_TAG.match(str(value).strip())
    return match.group(1).casefold() if match else ""


def _detect_language(text: str) -> str:
    """The code whose stopword list a text sample covers most fully.

    Too few tokens, a best candidate below a coverage floor, or a best candidate not
    clearly beating the runner-up returns empty, not a guess. CJK text is not split
    into words, so it reports nothing.
    """

    tokens = _LANGUAGE_WORD.findall(text.casefold())[:_LANGUAGE_SAMPLE_TOKENS]
    if len(tokens) < _LANGUAGE_MINIMUM_TOKENS:
        return ""
    present = set(tokens)
    scored: list[tuple[float, str]] = []
    for code in sorted(BM25_STOPWORD_LANGUAGES):
        stopwords = bm25_stopwords(code)
        if not stopwords:
            continue
        scored.append((len(present & stopwords) / len(stopwords), code))
    if not scored:
        return ""
    scored.sort(reverse=True)
    coverage, best = scored[0]
    runner_up = scored[1][0] if len(scored) > 1 else 0.0
    if (
        coverage < _LANGUAGE_MINIMUM_COVERAGE
        or coverage < runner_up * _LANGUAGE_MINIMUM_MARGIN
    ):
        return ""
    return best


def _base_metadata(
    *,
    source: SourceFile,
    document_id: str,
    digest: str,
    automatic: dict[str, Any],
    provenance: dict[str, str],
    warnings: list[str],
) -> dict[str, Any]:
    title = automatic.get("title") or source.path.stem
    title_source = provenance.get("title", "filename")
    authors = automatic.get("authors", [])
    author_source = provenance.get("authors", "missing")
    year = automatic.get("year")
    year_source = provenance.get("year", "missing")
    doi = automatic.get("doi", "")
    doi_source = provenance.get("doi", "missing")
    language = [str(code) for code in automatic.get("language") or [] if str(code)]
    language_source = provenance.get("language", "missing")
    resolved_authors = _normalize_text_list(list(authors or []))
    resolved_title = normalize_inline_text(str(title))
    if text_health_reasons(resolved_title):
        resolved_title = source.path.stem
        title_source = "filename"
        warnings.append("corrupt_extracted_title")
    clean_authors = [
        author for author in resolved_authors if not text_health_reasons(author)
    ]
    if len(clean_authors) != len(resolved_authors):
        warnings.append("corrupt_extracted_authors")
        author_source = "missing" if not clean_authors else author_source
    resolved_authors = clean_authors
    resolved_title = resolved_title or source.path.stem
    resolved_doi = _doi(str(doi)) or normalize_inline_text(str(doi or ""))
    metadata_warnings = list(dict.fromkeys(warnings))
    if title_source == "filename":
        metadata_warnings.append("title_from_filename")
    if not resolved_authors:
        metadata_warnings.append("authors_missing")
    metadata_provenance = {
        "title": title_source,
        "authors": author_source,
        "year": year_source,
        "doi": doi_source,
        "language": language_source,
        "categories": "missing",
        "keywords": "missing",
    }
    return {
        "document_id": document_id,
        "source_id": source.source_id,
        "source_path": source.project_relative_path,
        "source_relative_path": source.source_relative_path,
        "format": source.extension.removeprefix("."),
        "sha256": digest,
        "size": source.size,
        "mtime_ns": source.mtime_ns,
        "title": resolved_title,
        "authors": resolved_authors,
        "year": year,
        "doi": resolved_doi,
        "language": language,
        "categories": [],
        "keywords": [],
        "metadata_provenance": metadata_provenance,
        "metadata_warnings": list(dict.fromkeys(metadata_warnings)),
    }


def _pdf_author_list(metadata: dict[str, Any]) -> list[str]:
    author = normalize_inline_text(str(metadata.get("author") or ""))
    if not author:
        return []
    parts = [item.strip() for item in re.split(r"\s*[;|]\s*", author)]
    return [item for item in parts if _valid_author(item)]


def _pdf_metadata_identity(
    metadata: dict[str, Any],
    filename_stem: str,
    declared_language: str = "",
) -> tuple[dict[str, Any], dict[str, str], list[str]]:
    """Resolve PDF identity from the catalogue metadata supplied by Docling."""

    embedded_title = normalize_inline_text(str(metadata.get("title") or ""))
    authors = _pdf_author_list(metadata)
    language = _iso_language_code(declared_language)
    if _valid_title(embedded_title, filename_stem):
        title = embedded_title
        title_source = "pdf_metadata"
    else:
        title = filename_stem
        title_source = "filename"

    doi = ""
    doi_source = "missing"
    for candidate, source_name in (
        (embedded_title, "pdf_metadata"),
        (" ".join(str(value) for value in metadata.values()), "pdf_metadata"),
    ):
        if extracted := _doi(candidate):
            doi = extracted
            doi_source = source_name
            break

    year: int | None = None
    year_source = "missing"
    for field in ("creationDate", "modDate"):
        if match := _PDF_DATE_YEAR.search(
            normalize_inline_text(str(metadata.get(field) or ""))
        ):
            year = int(match.group(1))
            year_source = "pdf_metadata"
            break

    return (
        {
            "title": title,
            "authors": authors,
            "year": year,
            "doi": doi,
            "language": [language] if language else [],
        },
        {
            "title": title_source,
            "authors": "pdf_metadata" if authors else "missing",
            "year": year_source,
            "doi": doi_source,
            "language": "pdf_catalog" if language else "missing",
        },
        [],
    )


def _quality_flags(text: str, kind: str) -> list[str]:
    normalized = normalize_inline_text(text)
    alphabetic = sum(character.isalpha() for character in normalized)
    flags: list[str] = []
    if not has_searchable_alphanumeric_content(normalized) or (
        kind == "prose" and alphabetic < 3
    ):
        flags.append(EXTRACTION_ARTIFACT_TOKEN)
    return flags


def _epub_metadata_values(book: epub.EpubBook, name: str) -> list[str]:
    values: list[str] = []
    for value, _attributes in book.get_metadata("DC", name):
        normalized = normalize_inline_text(str(value))
        if normalized:
            values.append(normalized)
    return values


def _epub_visible_identity(
    book: epub.EpubBook,
    filename_stem: str,
) -> tuple[str, list[str]]:
    for spine_entry in book.spine[:5]:
        item_id = spine_entry[0] if isinstance(spine_entry, tuple) else spine_entry
        item = book.get_item_with_id(item_id)
        if item is None or item.get_type() != ebooklib.ITEM_DOCUMENT:
            continue
        soup = BeautifulSoup(item.get_content(), "html.parser")
        heading = soup.find(re.compile(r"^h1$")) or soup.find("title")
        visible_title = (
            normalize_inline_text(heading.get_text(" ", strip=True))
            if isinstance(heading, Tag)
            else ""
        )
        if not _valid_title(
            visible_title,
            filename_stem,
            reject_filename_match=False,
        ):
            visible_title = ""

        visible_authors: list[str] = []
        author_element = soup.find(
            attrs={"class": re.compile(r"(?:author|byline)", re.IGNORECASE)}
        )
        if not isinstance(author_element, Tag) and isinstance(heading, Tag):
            sibling = heading.find_next_sibling(["p", "div"])
            if isinstance(sibling, Tag) and _BYLINE.match(
                normalize_inline_text(sibling.get_text(" ", strip=True))
            ):
                author_element = sibling
        if isinstance(author_element, Tag):
            byline = _BYLINE.sub(
                "", normalize_inline_text(author_element.get_text(" ", strip=True))
            )
            visible_authors = [
                author
                for value in re.split(r"\s*(?:;|\band\b)\s*", byline)
                if (author := normalize_inline_text(value)) and _valid_author(author)
            ]
        if visible_title or visible_authors:
            return visible_title, visible_authors
    return "", []


def _epub_text_sample(book: epub.EpubBook, items: int = 3) -> str:
    parts: list[str] = []
    for spine_entry in book.spine[:items]:
        item_id = spine_entry[0] if isinstance(spine_entry, tuple) else spine_entry
        item = book.get_item_with_id(item_id)
        if item is None or item.get_type() != ebooklib.ITEM_DOCUMENT:
            continue
        soup = BeautifulSoup(item.get_content(), "html.parser")
        for unwanted in soup(["script", "style", "nav"]):
            unwanted.decompose()
        parts.append(soup.get_text(" ", strip=True))
    return "\n".join(parts)


def pdf_page_count(source: SourceFile) -> int:
    """Physical page count, after validating the PDF for extraction."""

    try:
        document = pymupdf.open(source.path)
    except Exception as exc:
        raise ExtractionError(f"Cannot open PDF source: {source.path}") from exc
    try:
        if document.needs_pass:
            raise ExtractionError(
                f"Password-protected PDF is unsupported: {source.path}"
            )
        return document.page_count
    finally:
        document.close()


# The book parsed last, kept while its file is unchanged. A build reads one spine
# item per checkpointed step, and parsing reads every item in the archive, so
# each step would otherwise parse the whole book again.
_PARSED_EPUB: tuple[tuple[str, int, int, int], epub.EpubBook] | None = None
_PARSED_EPUB_LOCK = threading.Lock()


def _read_epub(path: Path) -> epub.EpubBook:
    global _PARSED_EPUB
    try:
        stat = path.stat()
        identity = (str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        with _PARSED_EPUB_LOCK:
            if _PARSED_EPUB is not None and _PARSED_EPUB[0] == identity:
                return _PARSED_EPUB[1]
        book = epub.read_epub(str(path), options={"ignore_ncx": True})
    except Exception as exc:
        raise ExtractionError(f"Cannot open EPUB source: {path}") from exc
    with _PARSED_EPUB_LOCK:
        _PARSED_EPUB = (identity, book)
    return book


def prepare_epub_extraction(
    source: SourceFile,
    digest: str,
    *,
    book: epub.EpubBook | None = None,
) -> tuple[dict[str, Any], int]:
    """EPUB metadata and its deterministic spine work count."""

    if book is None:
        book = _read_epub(source.path)
    titles = _epub_metadata_values(book, "title")
    authors = _epub_metadata_values(book, "creator")
    identifiers = _epub_metadata_values(book, "identifier")
    dates = _epub_metadata_values(book, "date")
    visible_title, visible_authors = _epub_visible_identity(book, source.path.stem)
    opf_title = next(
        (item for item in titles if _valid_title(item, source.path.stem)),
        "",
    )
    title = opf_title or visible_title or source.path.stem
    valid_opf_authors = [author for author in authors if _valid_author(author)]
    resolved_authors = valid_opf_authors or visible_authors
    year_match = next(
        (_YEAR.search(item) for item in dates if _YEAR.search(item)),
        None,
    )
    declared_language = next(
        (
            code
            for item in _epub_metadata_values(book, "language")
            if (code := _iso_language_code(item))
        ),
        "",
    )
    language = declared_language or _detect_language(_epub_text_sample(book))
    automatic = {
        "title": title,
        "authors": resolved_authors,
        "year": int(year_match.group(1)) if year_match else None,
        "doi": next((_doi(item) for item in identifiers if _doi(item)), ""),
        "language": [language] if language else [],
    }
    title_source = (
        "epub_opf" if opf_title else "epub_visible" if visible_title else "filename"
    )
    record = _base_metadata(
        source=source,
        document_id=_document_id(source, digest),
        digest=digest,
        automatic=automatic,
        provenance={
            "title": title_source,
            "authors": (
                "epub_opf"
                if valid_opf_authors
                else "epub_visible"
                if visible_authors
                else "missing"
            ),
            "year": "epub_opf" if year_match else "missing",
            "doi": "epub_opf" if automatic["doi"] else "missing",
            "language": (
                "epub_opf"
                if declared_language
                else "text_sample"
                if language
                else "missing"
            ),
        },
        warnings=(
            ["conflicting_candidates"]
            if opf_title
            and visible_title
            and opf_title.casefold() != visible_title.casefold()
            else []
        ),
    )
    record.update(
        {
            "spine_items": len(book.spine),
            "extracted_units": 0,
            "empty_units": 0,
        }
    )
    return record, len(book.spine)


def extract_mobi(
    source: SourceFile,
    digest: str | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Read MOBI once without writing conversions; apply the EPUB HTML rules.

    MOBI7 uses the original HTML stream. KF8 uses reconstructed text parts. These
    logical sections and element paths are not printed pages or EPUB source paths.
    """
    try:
        return extract_mobi_payload(
            source.path,
            {
                "source": {
                    "source_id": source.source_id,
                    "source_relative_path": source.source_relative_path,
                    "project_relative_path": source.project_relative_path,
                    "extension": source.extension,
                    "size": source.size,
                    "mtime_ns": source.mtime_ns,
                },
                "digest": digest or sha256_file(source.path),
            },
        )
    except MobiReadError as exc:
        raise ExtractionError(f"Cannot open MOBI source {source.path}: {exc}") from exc


def _mobi_units_from_text(
    source: SourceFile, digest: str, parsed: MobiText
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Worker-only HTML conversion; the serving parent never parses MOBI HTML."""
    book = epub.EpubBook()
    for mobi_key, dc_key in (
        ("Title", "title"),
        ("Creator", "creator"),
        ("Published", "date"),
        ("Language", "language"),
    ):
        for value in parsed.metadata.get(mobi_key, []):
            book.add_metadata("DC", dc_key, value)
    items = []
    for index, html in enumerate(parsed.parts):
        item = epub.EpubHtml(
            uid=f"part-{index + 1}", file_name=f"part-{index + 1:06d}.html"
        )
        item.set_content(html.encode("utf-8"))
        book.add_item(item)
        items.append(item)
    book.spine = [(item.id, "yes") for item in items]
    record, _total = prepare_epub_extraction(source, digest, book=book)
    record["metadata_provenance"] = {
        key: value.replace("epub_opf", "mobi_header").replace(
            "epub_visible", "mobi_visible"
        )
        for key, value in record["metadata_provenance"].items()
    }
    record.update(mobi_version=parsed.version, mobi_compression=parsed.compression)
    units: list[dict[str, Any]] = []
    removals = empty_removal_counts()
    empty = 0
    for index, item in enumerate(items):
        batch, absent, counts = _extract_html_item(source, record, index, item)
        units.extend(batch)
        empty += int(absent)
        removals = merge_removal_counts(removals, counts)
    record.update(removals)
    record.update(extracted_units=len(units), empty_units=empty)
    if not units:
        raise ExtractionError(_no_text_layer_message(source, "MOBI"))
    return record, units


def _epub_fragment(tag: Tag) -> tuple[str, str] | None:
    """An existing XHTML fragment and its source attribute, verbatim."""

    for attribute in ("id", "name"):
        value = tag.get(attribute)
        if value is None:
            continue
        fragment = (
            " ".join(str(item) for item in value)
            if isinstance(value, list)
            else str(value)
        )
        if fragment.strip():
            return fragment, attribute
    return None


def _epub_element_path(tag: Tag) -> str:
    """A deterministic, human-inspectable path within one XHTML item."""

    parts: list[str] = []
    current: Tag | None = tag
    while isinstance(current, Tag) and current.name != "[document]":
        name = str(current.name).casefold()
        parent = current.parent
        same_name_index = 1
        if isinstance(parent, Tag):
            same_name_index = 0
            for sibling in parent.children:
                if isinstance(sibling, Tag) and str(sibling.name).casefold() == name:
                    same_name_index += 1
                if sibling is current:
                    break
        parts.append(f"{name}[{same_name_index}]")
        current = parent if isinstance(parent, Tag) else None
    return "/" + "/".join(reversed(parts))


def _epub_block_name(tag: Tag) -> str:
    """The element a tag stands for when the reader divides text into blocks.

    Some journal EPUBs print every paragraph as `<div role="paragraph">` and
    contain no `<p>` at all, so an element with that role is read as the
    paragraph it declares itself to be.
    """

    name = str(tag.name).casefold()
    if str(tag.get("role") or "").casefold() == "paragraph":
        return "p"
    return name


def _epub_previous_sibling(tag: Tag) -> Tag | None:
    """The element printed just before a tag, ignoring whitespace and comments."""

    sibling = tag.previous_sibling
    while isinstance(sibling, NavigableString) and (
        isinstance(sibling, Comment) or not sibling.strip()
    ):
        sibling = sibling.previous_sibling
    return sibling if isinstance(sibling, Tag) else None


def _is_empty_epub_anchor(tag: Tag | None) -> bool:
    return (
        tag is not None
        and _epub_fragment(tag) is not None
        and tag.find(True) is None
        and not tag.get_text(strip=True)
    )


def _epub_marker_extent(link: Tag, block: Tag) -> Tag:
    """The link, or the inline wrappers around it that print nothing else."""

    extent = link
    marker = normalize_inline_text(link.get_text())
    while (
        isinstance(extent.parent, Tag)
        and extent.parent is not block
        and str(extent.parent.name).casefold() in _EPUB_MARKER_WRAPPERS
        and normalize_inline_text(extent.parent.get_text()) == marker
    ):
        extent = extent.parent
    return extent


def _is_epub_noteref(link: Tag, block: Tag) -> bool:
    """Whether a link is a note reference rather than linked prose.

    The link points at a fragment and prints only a marker, and its markup says it
    is a reference: the publisher's `noteref` type, a superscript, or an anchor the
    note links back to, on the link or printed just before it. A cross-reference
    such as "chapter 3" prints the same digit but carries none of those, so it
    keeps its text.
    """

    if "#" not in str(link.get("href") or ""):
        return False
    if not _EPUB_NOTE_MARKER_TEXT.fullmatch(normalize_inline_text(link.get_text())):
        return False
    if "noteref" in str(link.get("epub:type") or "").casefold().split():
        return True
    if link.find("sup") is not None or link.find_parent("sup") is not None:
        return True
    if _epub_fragment(link) is not None:
        return True
    return _is_empty_epub_anchor(
        _epub_previous_sibling(_epub_marker_extent(link, block))
    )


def _drop_epub_noterefs(block: Tag) -> int:
    """Remove the note references that follow text in a block; return the count.

    A reference is a pointer to a note, not evidence. Its anchor is where the note
    links back to, so leaving it would split the sentence around it into two units
    and print the marker at the start of the second. The empty anchor printed just
    before a reference is that same way back, and it goes with the reference. A
    marker that opens its block is a note's own number in the notes section, and
    it stays.
    """

    references: list[Tag] = []
    seen_text = False
    for node in block.descendants:
        if isinstance(node, Comment):
            continue
        if isinstance(node, NavigableString):
            seen_text = seen_text or bool(node.strip())
        elif (
            seen_text
            and str(node.name).casefold() == "a"
            and _is_epub_noteref(node, block)
        ):
            references.append(node)
    for link in references:
        extent = _epub_marker_extent(link, block)
        previous = _epub_previous_sibling(extent)
        if previous is not None and _is_empty_epub_anchor(previous):
            previous.decompose()
        extent.decompose()
    return len(references)


def _epub_semantic_blocks(
    soup: BeautifulSoup,
) -> list[tuple[Tag, tuple[str, str] | None]]:
    """Outermost semantic blocks and a nearest standalone anchor."""

    tags = list(soup.find_all(True))
    positions = {id(tag): index for index, tag in enumerate(tags)}
    blocks = [
        tag
        for tag in tags
        if _epub_block_name(tag) in _EPUB_SEMANTIC_ELEMENTS
        and not any(
            isinstance(parent, Tag)
            and _epub_block_name(parent) in _EPUB_SEMANTIC_ELEMENTS
            for parent in tag.parents
        )
    ]
    block_ids = {id(tag) for tag in blocks}
    standalone_anchors: list[tuple[int, tuple[str, str]]] = []
    for tag in tags:
        anchor = _epub_fragment(tag)
        if anchor is None or id(tag) in block_ids:
            continue
        if any(id(parent) in block_ids for parent in tag.parents):
            continue
        standalone_anchors.append((positions[id(tag)], anchor))

    results: list[tuple[Tag, tuple[str, str] | None]] = []
    anchor_index = 0
    pending_anchor: tuple[str, str] | None = None
    for block in blocks:
        block_position = positions[id(block)]
        while (
            anchor_index < len(standalone_anchors)
            and standalone_anchors[anchor_index][0] < block_position
        ):
            pending_anchor = standalone_anchors[anchor_index][1]
            anchor_index += 1
        results.append((block, pending_anchor))
        pending_anchor = None
    return results


def _epub_text_segments(
    element: Tag,
    initial_anchor: tuple[str, str, str] | None,
) -> list[tuple[str, tuple[str, str, str] | None]]:
    """Split visible block text at nested XHTML anchors, leaking no marker."""

    segments: list[tuple[str, tuple[str, str, str] | None]] = []
    parts: list[str] = []
    active_anchor = initial_anchor

    def flush() -> None:
        text = normalize_reading_text("".join(parts))
        parts.clear()
        if text:
            segments.append((text, active_anchor))

    def inside(node: PageElement, block: Tag) -> bool:
        # Identity, not `==`: a Tag compares equal to any tag with the same markup.
        parent = node.parent
        while parent is not None and parent is not element:
            if parent is block:
                return True
            parent = parent.parent
        return False

    # A nested block ends a paragraph where it closes as well as where it opens,
    # so text after a quotation inside a paragraph is not run into its last word.
    open_blocks: list[Tag] = []
    for node in element.descendants:
        while open_blocks and not inside(node, open_blocks[-1]):
            open_blocks.pop()
            parts.append("\n\n")
        if isinstance(node, Comment):
            continue
        if isinstance(node, Tag):
            anchor = _epub_fragment(node)
            if anchor is not None:
                flush()
                active_anchor = (anchor[0], "nested", anchor[1])
            name = _epub_block_name(node)
            if name == "br":
                parts.append("\n")
            elif name in _EPUB_CONTENT_ELEMENTS:
                parts.append("\n\n")
                open_blocks.append(node)
            continue
        if isinstance(node, NavigableString):
            parts.append(str(node))
    flush()
    return segments


def _epub_table_text(table: Tag) -> str:
    rows: list[str] = []
    for row in table.find_all("tr"):
        cells = [
            normalize_inline_text(cell.get_text(" ", strip=True))
            for cell in row.find_all(["th", "td"])
        ]
        if any(cells):
            rows.append(" | ".join(cells))
    return "\n".join(rows)


_EPUB_SKIPPED_ELEMENTS = (
    "script",
    "style",
    "nav",
    "aside",
)
# An EPUB has no page geometry, so furniture is named rather than measured. These
# are the class and id fragments publishers put on marginal notes, sidebars, and
# note apparatus; nothing here is a word that could occur in body text.
_EPUB_FURNITURE_MARKERS = re.compile(
    r"(?:^|[\s_-])(?:sidebars?|sidenotes?|margin-?notes?|footnotes?|"
    r"endnotes?|page-?breaks?|adverts?|newsletters?)(?:[\s_-]|$)",
    re.IGNORECASE,
)
_EPUB_FURNITURE_ROLES = frozenset(
    {"complementary", "doc-endnotes", "doc-footnote", "doc-noteref", "note"}
)


def _is_epub_furniture(tag: Tag) -> bool:
    """Whether an XHTML element is furniture the document did not argue with.

    Only the element's own name, role, class, and id are consulted, so a
    sentence that mentions footnotes inside an ordinary paragraph is untouched:
    the rule reads markup, not prose.
    """

    if str(tag.get("role") or "").casefold() in _EPUB_FURNITURE_ROLES:
        return True
    for attribute in ("class", "id"):
        value = tag.get(attribute)
        if value is None:
            continue
        joined = (
            " ".join(str(item) for item in value)
            if isinstance(value, list)
            else str(value)
        )
        if _EPUB_FURNITURE_MARKERS.search(joined):
            return True
    return False


def _epub_locator(
    *,
    section_index: int,
    section_title: str,
    href: str,
    element: Tag,
    block_index: int,
    anchor: tuple[str, str, str] | None,
) -> dict[str, Any]:
    locator: dict[str, Any] = {
        "type": "epub_section",
        "section_index": section_index,
        "section_title": section_title,
        "href": href,
        "element_path": _epub_element_path(element),
        "block_index": block_index,
    }
    if anchor is not None:
        fragment, relation, source = anchor
        locator.update(
            {
                "fragment": fragment,
                "href_with_fragment": f"{href}#{fragment}",
                "anchor_relation": relation,
                "anchor_source": source,
            }
        )
    return locator


def extract_epub_spine_item(
    source: SourceFile,
    document_record: dict[str, Any],
    spine_index: int,
) -> tuple[list[dict[str, Any]], bool, dict[str, int]]:
    """One EPUB spine entry, with its original locator preserved.

    The returned counts name the elements this entry gave up, so a source that
    carried sidebars says so rather than reporting nothing. A spine entry the
    reader is told is furniture and that has none of it reports zeros, which is
    what a real collection without sidebars must show.
    """

    removals = empty_removal_counts()
    book = _read_epub(source.path)
    spine_entry = book.spine[spine_index]
    item_id = spine_entry[0] if isinstance(spine_entry, tuple) else spine_entry
    item = book.get_item_with_id(item_id)
    if item is None or item.get_type() != ebooklib.ITEM_DOCUMENT:
        return [], False, removals
    return _extract_html_item(source, document_record, spine_index, item)


def _extract_html_item(
    source: SourceFile,
    document_record: dict[str, Any],
    spine_index: int,
    item: epub.EpubHtml,
) -> tuple[list[dict[str, Any]], bool, dict[str, int]]:
    """Shared reflowable HTML cleanup; locators keep the source format distinct."""
    removals = empty_removal_counts()
    soup = BeautifulSoup(item.get_content(), "html.parser")
    # The navigation, scripts, and stylesheet are removed on every EPUB and say nothing
    # about the source's own furniture, so they are not counted. The count is what a
    # reader uses to decide whether the cleanup did anything.
    _EPUB_STRUCTURAL_ELEMENTS = frozenset({"script", "style", "nav"})
    outermost = [
        tag
        for tag in soup.find_all(True)
        if str(tag.name).casefold() not in _EPUB_STRUCTURAL_ELEMENTS
        and (
            str(tag.name).casefold() in _EPUB_SKIPPED_ELEMENTS
            or _is_epub_furniture(tag)
        )
        and not any(
            parent is not soup
            and (
                str(parent.name).casefold() in _EPUB_SKIPPED_ELEMENTS
                or _is_epub_furniture(parent)
            )
            for parent in tag.parents
        )
    ]
    for unwanted in soup(list(_EPUB_SKIPPED_ELEMENTS)):
        unwanted.decompose()
    for furniture in outermost:
        furniture.decompose()
        removals["removed_epub_furniture_elements"] += 1
    heading = soup.find(re.compile(r"^h[1-6]$"))
    section_title = (
        normalize_inline_text(heading.get_text(" ", strip=True))
        if heading is not None
        else ""
    )
    units: list[dict[str, Any]] = []
    document_id = str(document_record["document_id"])
    source_id = str(document_record["source_id"])
    title = str(document_record["title"])
    href = str(item.get_name() or "")
    block_index = 0
    pending_heading_parts: list[str] = []
    pending_heading_anchor: tuple[str, str, str] | None = None
    pending_heading_element: Tag | None = None

    def append_unit(
        text: str,
        *,
        element: Tag,
        anchor: tuple[str, str, str] | None,
        content_kind: str,
    ) -> None:
        nonlocal block_index
        if not text.strip():
            return
        block_index += 1
        locator = _epub_locator(
            section_index=spine_index + 1,
            section_title=section_title,
            href=href,
            element=element,
            block_index=block_index,
            anchor=anchor,
        )
        source_format = source.extension.removeprefix(".")
        if source_format == "mobi":
            locator["type"] = "mobi_section"
        units.append(
            {
                "id": (
                    f"{document_id}:{source_format}-section:{spine_index + 1:06d}:"
                    f"block:{block_index:06d}"
                ),
                "document_id": document_id,
                "source_id": source_id,
                "title": title,
                "contents": text,
                "content_kind": content_kind,
                "annotations": [],
                "quality_flags": _quality_flags(text, content_kind),
                "locator": locator,
            }
        )

    for element, preceding_fragment in _epub_semantic_blocks(soup):
        element_name = _epub_block_name(element)
        if element_name != "table":
            removals["removed_epub_furniture_elements"] += _drop_epub_noterefs(element)
        element_fragment = _epub_fragment(element)
        initial_anchor = (
            (element_fragment[0], "element", element_fragment[1])
            if element_fragment is not None
            else (
                (preceding_fragment[0], "preceding", preceding_fragment[1])
                if preceding_fragment is not None
                else None
            )
        )
        if element_name in _EPUB_HEADING_ELEMENTS:
            heading_segments = _epub_text_segments(element, initial_anchor)
            if source.extension == ".mobi":
                section_title = normalize_inline_text(element.get_text(" ", strip=True))
            pending_heading_parts.extend(text for text, _anchor in heading_segments)
            if pending_heading_anchor is None:
                pending_heading_anchor = next(
                    (
                        anchor
                        for _text, anchor in heading_segments
                        if anchor is not None
                    ),
                    initial_anchor,
                )
            pending_heading_element = element
            continue

        if element_name == "table":
            text = _epub_table_text(element)
            if not text:
                continue
            anchor = initial_anchor
            if anchor is None:
                nested_anchor = next(
                    (
                        found
                        for descendant in element.find_all(True)
                        if (found := _epub_fragment(descendant)) is not None
                    ),
                    None,
                )
                if nested_anchor is not None:
                    anchor = (nested_anchor[0], "nested", nested_anchor[1])
            if anchor is None and pending_heading_anchor is not None:
                anchor = (
                    pending_heading_anchor[0],
                    "preceding",
                    pending_heading_anchor[2],
                )
            pending_heading_parts.clear()
            pending_heading_anchor = None
            pending_heading_element = None
            append_unit(text, element=element, anchor=anchor, content_kind="table")
            continue

        segments = _epub_text_segments(element, initial_anchor)
        if not segments:
            continue
        if pending_heading_parts:
            first_text, first_anchor = segments[0]
            segments[0] = (
                "\n\n".join([*pending_heading_parts, first_text]),
                first_anchor
                or (
                    (
                        pending_heading_anchor[0],
                        "preceding",
                        pending_heading_anchor[2],
                    )
                    if pending_heading_anchor is not None
                    else None
                ),
            )
        pending_heading_parts.clear()
        pending_heading_anchor = None
        pending_heading_element = None
        for text, anchor in segments:
            append_unit(
                text,
                element=element,
                anchor=anchor,
                content_kind="prose",
            )

    if pending_heading_parts and pending_heading_element is not None:
        append_unit(
            "\n\n".join(pending_heading_parts),
            element=pending_heading_element,
            anchor=pending_heading_anchor,
            content_kind="prose",
        )
    return units, not units, removals


def _readable_epub_unit_text(text: str) -> str | None:
    """Preserve EPUB's existing majority-readable unit cleanup."""

    if not text_corruption_reasons(text):
        return None
    readable = "\n\n".join(
        paragraph
        for paragraph in text.split("\n\n")
        if paragraph.strip() and not text_corruption_reasons(paragraph)
    )
    if not readable or text_health_reasons(readable):
        return None
    if len(readable) * 100 < len(text) * 50:
        return None
    return readable


def screen_source_units(
    source: SourceFile,
    document: dict[str, Any],
    units: list[dict[str, Any]],
    *,
    removals: dict[str, int] | None = None,
    maximum_unclean_percent: float | None = None,
) -> list[dict[str, Any]]:
    """Repair PDF passages locally, then enforce their document-wide loss budget.

    For PDFs, `maximum_unclean_percent` caps irreversible substantive-character
    loss over the document after confirmed furniture and non-evidence are omitted.
    Clean paragraphs bypass repair. Damaged tokens leave visible gaps, not guessed
    letters or silently joined words. EPUB retains its existing screening policy.

    A paragraph that carries no alphanumeric content and no corruption evidence is
    non-evidence: it is omitted and recorded on `excluded_symbol_only_passages`,
    never on the corrupt counters. A documented formatting glyph, such as a symbol
    font's bullet, is not corruption either, so it does not withhold its passage.

    This is the one gate both extraction paths apply, and it applies the section
    removal as well: a staged build writes a page batch to disk before it knows
    what the whole file looks like, so it cannot decide at batch time whether a
    heading on page 40 opens a list that runs to page 43. Deciding here, over every
    unit the source produced, is what makes the two paths reach the same answer.

    Returns the retained units and records every exclusion and removal on the
    document. Raises `ExtractionError` with the file's path, its reasons, and a
    remedy when nothing survives, which is what makes the ingest that found it
    abort rather than publish a generation holding one unreadable file.
    """

    cleaned, section_counts = strip_non_argument_units(units)
    passage_level = source.extension == ".pdf"

    retained: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    rejected_passages: list[dict[str, Any]] = []
    repaired_passages: list[dict[str, Any]] = []
    omitted_symbol_only_units: list[dict[str, Any]] = []
    omitted_symbol_only_passages: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    kept_characters = 0
    withheld_characters = 0
    cleaned_corrupt_spans = 0
    partially_cleaned_passages = 0
    for unit in cleaned:
        text = str(unit.get("contents") or "")
        if not passage_level:
            # EPUB is outside the PDF recovery change. Preserve its unit and
            # source-level acceptance rules, including the majority guard.
            kept_characters += len(text)
            reasons = text_health_reasons(text)
            if not reasons:
                retained.append(unit)
            elif (readable_epub := _readable_epub_unit_text(text)) is not None:
                retained.append({**unit, "contents": readable_epub})
                withheld_characters += len(text) - len(readable_epub)
            else:
                withheld_characters += len(text)
                reason_counts.update(reasons)
                rejected.append(
                    {
                        "unit_id": str(unit.get("id") or ""),
                        "locator": dict(unit.get("locator") or {}),
                        "reasons": reasons,
                    }
                )
            continue
        readable: list[str] = []
        unit_reasons: list[str] = []
        unit_symbol_only = False
        for passage_index, paragraph in enumerate(text.split("\n\n")):
            if not paragraph.strip():
                continue
            reasons = passage_health_reasons(paragraph, maximum_unclean_percent=0.0)
            if not reasons:
                readable.append(paragraph)
                kept_characters += sum(
                    not character.isspace() for character in paragraph
                )
                continue
            decision = {
                "unit_id": str(unit.get("id") or ""),
                "passage_index": passage_index,
                "locator": dict(unit.get("locator") or {}),
                "reasons": reasons,
            }
            if set(reasons) == {"symbol_only"}:
                # A punctuation-only fragment carries no searchable content and
                # no corruption evidence. It is non-evidence and omitted without
                # a corruption verdict, so it never reaches the corrupt counters.
                omitted_symbol_only_passages.append(decision)
                unit_symbol_only = True
                continue
            repaired = clean_unclean_passage(paragraph)
            remaining = passage_health_reasons(repaired, maximum_unclean_percent=0.0)
            if repaired.strip() and not remaining:
                readable.append(repaired)
                repaired_passages.append(decision)
                kept_characters += sum(
                    not character.isspace() for character in repaired
                )
                continue
            if set(remaining) == {"symbol_only"}:
                omitted_symbol_only_passages.append({**decision, "reasons": remaining})
                unit_symbol_only = True
                continue
            meaningful_size = sum(not character.isspace() for character in repaired)
            kept_characters += meaningful_size
            local_text, local_loss, spans = clean_damaged_pdf_passage(repaired)
            if has_searchable_alphanumeric_content(
                local_text
            ) and not passage_health_reasons(local_text, maximum_unclean_percent=0.0):
                readable.append(local_text)
                withheld_characters += local_loss
                cleaned_corrupt_spans += len(spans)
                partially_cleaned_passages += 1
                repaired_passages.append(
                    {**decision, "lost_characters": local_loss, "removed_spans": spans}
                )
                continue
            withheld_characters += meaningful_size
            excluded_reasons = remaining or reasons
            reason_counts.update(excluded_reasons)
            unit_reasons.extend(excluded_reasons)
            rejected_passages.append({**decision, "reasons": excluded_reasons})
        if readable:
            retained.append({**unit, "contents": "\n\n".join(readable)})
        elif unit_reasons:
            rejected.append(
                {
                    "unit_id": str(unit.get("id") or ""),
                    "locator": dict(unit.get("locator") or {}),
                    "reasons": list(dict.fromkeys(unit_reasons)),
                }
            )
        elif unit_symbol_only:
            omitted_symbol_only_units.append(
                {
                    "unit_id": str(unit.get("id") or ""),
                    "locator": dict(unit.get("locator") or {}),
                    "reasons": ["symbol_only"],
                }
            )
        else:
            rejected.append(
                {
                    "unit_id": str(unit.get("id") or ""),
                    "locator": dict(unit.get("locator") or {}),
                    "reasons": [],
                }
            )
    # A direct build has already written its counters onto the document and passes
    # nothing here; a staged build holds them in its checkpoint and passes them in.
    # Either way the section removal this gate performs is added to both.
    counted = merge_removal_counts(
        removals
        if removals is not None
        else {
            key: document[key]
            for key in REMOVAL_FIELDS
            if document.get(key) is not None
        },
        section_counts,
    )
    document.update(counted)
    document["excluded_corrupt_unit_count"] = len(rejected)
    document["excluded_corrupt_units"] = rejected
    document["excluded_corrupt_passage_count"] = len(rejected_passages)
    document["excluded_corrupt_passages"] = rejected_passages
    document["excluded_symbol_only_unit_count"] = len(omitted_symbol_only_units)
    document["excluded_symbol_only_units"] = omitted_symbol_only_units
    document["excluded_symbol_only_passage_count"] = len(omitted_symbol_only_passages)
    document["excluded_symbol_only_passages"] = omitted_symbol_only_passages
    document["cleaned_passage_count"] = len(repaired_passages)
    document["cleaned_passages"] = repaired_passages
    if passage_level:
        document["substantive_character_count"] = kept_characters
        document["discarded_corrupt_character_count"] = withheld_characters
        document["cleaned_corrupt_span_count"] = cleaned_corrupt_spans
        document["partially_cleaned_passage_count"] = partially_cleaned_passages
    # Measured here and nowhere else, so a source extracted before it was measured
    # carries no figure rather than a guessed one.
    document["unclean_character_rate"] = (
        round(withheld_characters / kept_characters, 6) if kept_characters else 0.0
    )
    document["extracted_units"] = len(retained)
    if rejected_passages or rejected or cleaned_corrupt_spans:
        document["metadata_warnings"] = list(
            dict.fromkeys(
                [
                    *document.get("metadata_warnings", []),
                    *(
                        ["corrupt_extraction_spans_excluded"]
                        if cleaned_corrupt_spans
                        else []
                    ),
                    *(
                        ["corrupt_extraction_passages_excluded"]
                        if rejected_passages
                        else []
                    ),
                    *(["corrupt_extraction_units_excluded"] if rejected else []),
                ]
            )
        )

    fatal = source_health_reasons(
        unit_count=len(cleaned),
        retained_count=len(retained),
        withheld_reasons=reason_counts,
        kept_characters=kept_characters,
        removed_characters=int(
            (counted or {}).get("removed_non_argument_characters") or 0
        ),
        image_only_pages=int((counted or {}).get("image_only_pages") or 0),
        physical_pages=int(document.get("physical_pages") or 0),
        letter_characters=sum(
            character.isalpha()
            for unit in retained
            for character in str(unit.get("contents") or "")
        ),
        withheld_characters=withheld_characters,
        maximum_unclean_percent=maximum_unclean_percent,
        passage_level=passage_level,
    )
    if fatal:
        error_type = (
            UncleanSourceError if SOURCE_REASON_UNCLEAN in fatal else ExtractionError
        )
        # The refusal names where the unreadable material sat. A punctuation-only
        # unit is not corrupt, but it is why nothing survived, so it is reported
        # here for its locator and its reason without entering the corrupt counts.
        reported_units = [*rejected, *omitted_symbol_only_units]
        if passage_level:
            reported_units.extend(
                decision
                for decision in repaired_passages
                if decision.get("lost_characters")
            )
        reported_counts = Counter(reason_counts)
        reported_counts["symbol_only"] += len(omitted_symbol_only_units)
        raise error_type(
            _refusal_message(
                source,
                fatal,
                reported_units,
                reported_counts,
                loss=(withheld_characters, kept_characters, maximum_unclean_percent)
                if passage_level
                else None,
            )
        )
    if document.get("image_only_pages"):
        document["metadata_warnings"] = list(
            dict.fromkeys(
                [
                    *document.get("metadata_warnings", []),
                    "image_only_pages_present",
                ]
            )
        )
    if passage_level and document.get("metadata_warnings"):
        document["metadata_warnings"] = sorted(
            set(document.get("metadata_warnings", []))
        )
    return retained


def strip_non_argument_units(
    units: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Remove confirmed apparatus across a whole source's units.

    A reference list does not fit in one unit. A book's may open with a heading on
    its own page and run over the next three, and an EPUB emits its heading as one
    element and each entry as another, so no single unit ever holds a heading and
    its entries together. Deciding per unit therefore removes nothing on exactly
    the sources that need it most, which is why this runs here, over the whole
    unit sequence in reading order, rather than inside a page or spine item.

    A unit that ends up with no paragraph left is dropped, so a page holding
    nothing but entries contributes no unit at all. The counters say what went:
    how many reference paragraphs, how many note paragraphs, and how many
    characters.

    Docling units carry their original structural labels. Their headings and
    reference entries guide the decision; other item kinds preserve substantive
    captions, formulas, and notes. EPUB and MOBI keep their markup/text rules.
    """

    units_by_index = list(enumerate(units))
    paragraphs: list[str] = []
    positions: list[tuple[int, int]] = []
    labels: list[str] = []
    structured = any("docling_label" in unit for unit in units)
    for unit_index, unit in units_by_index:
        for offset, paragraph in enumerate(
            str(unit.get("contents") or "").split("\n\n")
        ):
            if not paragraph.strip():
                continue
            paragraphs.append(paragraph)
            positions.append((unit_index, offset))
            labels.append(str(unit.get("docling_label") or ""))

    removed_kinds = non_argument_removal_flags(
        paragraphs, labels=labels if structured else None
    )
    surviving: dict[int, list[str]] = {index: [] for index, _unit in units_by_index}
    removed_reference = 0
    removed_notes = 0
    removed_characters = 0
    for (unit_index, _offset), paragraph, kind in zip(
        positions, paragraphs, removed_kinds
    ):
        if kind is None:
            surviving[unit_index].append(paragraph)
            continue
        removed_characters += len(paragraph)
        if kind == "footnotes":
            removed_notes += 1
        else:
            removed_reference += 1

    cleaned: list[dict[str, Any]] = []
    for unit_index, unit in units_by_index:
        text = "\n\n".join(surviving[unit_index])
        if not text:
            removed_characters += len(str(unit.get("contents") or ""))
            continue
        cleaned.append({**unit, "contents": text})

    counts = merge_removal_counts(
        None,
        {
            "removed_reference_segments": removed_reference,
            "removed_note_segments": removed_notes,
            "removed_non_argument_characters": removed_characters,
        },
    )
    return cleaned, counts


def _no_text_layer_message(source: SourceFile, label: str) -> str:
    """What a file with no text layer says, and what is not done about it.

    A scanned page has no text to read, and this app does not recognise one. The
    message says that plainly rather than implying OCR might run, and it names the
    two things a reader can do.
    """

    return (
        f"{label} yielded no text layer ({SOURCE_REASON_NO_TEXT_LAYER}): "
        f"{source.path}. Every page carried an image or a drawing and no "
        "extractable text. No OCR is performed, so nothing here can be read. "
        "Run OCR yourself outside research-rag and supply a text-layer PDF, "
        "or supply an EPUB, or exclude "
        f"the source with `research-rag exclude {source.source_relative_path}"
        ' --reason "no text layer"`.'
    )


def _refusal_message(
    source: SourceFile,
    reasons: list[str],
    rejected: list[dict[str, Any]],
    reason_counts: Counter[str],
    *,
    loss: tuple[int, int, float | None] | None = None,
) -> str:
    """What a refused source says, in a form a reader can act on.

    A source with nothing readable left keeps its established opening sentence,
    because that is what a reader of a failure record looks for. What follows it
    is what makes the record actionable: the reasons the file's own excluded
    units carry, where the first of them sits, and the two remedies a person has
    — exclude the file, or open it. No unit text is quoted, because a source can
    be refused for text that is unreadable and unsafe to repeat.
    """

    lead = (
        "Source produced no readable English-oriented text after unhealthy "
        "extraction units were excluded"
        if reasons == [SOURCE_REASON_NO_TEXT]
        else f"Source refused ({', '.join(reasons)})"
    )
    if SOURCE_REASON_NO_TEXT_LAYER in reasons:
        return (
            _no_text_layer_message(source, "Source")
            + f" Refused for: {', '.join(reasons)}."
        )
    if SOURCE_REASON_UNCLEAN in reasons:
        if source.extension == ".pdf" and loss is not None:
            lost, total, limit = loss
            percent = 100 * lost / total if total else 100.0
            location = ""
            if rejected:
                locator = rejected[0].get("locator") or {}
                if locator.get("page"):
                    location = f" First loss at physical PDF page {locator['page']}."
            return (
                f"PDF substantive-text loss exceeds the document budget: {source.path}. "
                f"Lost {lost} of {total} substantive characters ({percent:.2f}%), "
                f"above ingestion.maximum_unclean_percent={limit:g}.{location} "
                "This source is omitted from the validated partial generation; "
                "the original and selected generation are unchanged. Repair or "
                "replace the text layer, or explicitly review the loss budget."
            )
        return (
            f"Source is not clean ({', '.join(reasons)}): {source.path}. "
            f"No readable text was recovered within this project's {source.extension[1:].upper()} loss limit. "
            "The file is never edited. Supply a readable ebook, or exclude it with "
            f"`research-rag exclude {source.source_relative_path} --reason "
            '"not clean"`, or raise ingestion.maximum_unclean_percent.'
        )
    counts = ", ".join(
        f"{reason}={count}" for reason, count in sorted(reason_counts.items())
    )
    location = ""
    if rejected:
        first = rejected[0]["locator"]
        if first.get("type") == "pdf_page":
            location = f" First excluded unit at page {first.get('page')}."
        elif first.get("type") in {"epub_section", "mobi_section"}:
            location = (
                f" First excluded unit at section {first.get('section_index')}"
                f" ({first.get('href')})."
            )
    detail = f" Reasons on excluded units: {counts}." if counts else ""
    return (
        f"{lead}: {source.path}.{location}{detail}"
        " Automatic text extraction and cleaning could not recover these passages."
        " The file is never edited. If it needs OCR, run that outside research-rag."
        f" Exclude it with `research-rag exclude {source.source_relative_path}"
        ' --reason "text could not be read"`, or open the file and check it.'
    )


def _docling_document_record(
    source: SourceFile,
    digest: str,
    pdf_metadata: dict[str, Any],
    total_pages: int,
    docling_env: DoclingEnv,
) -> dict[str, Any]:
    declared_language = str(pdf_metadata.get("language") or "")
    automatic, provenance, warnings = _pdf_metadata_identity(
        pdf_metadata,
        source.path.stem,
        declared_language=declared_language,
    )
    record = _base_metadata(
        source=source,
        document_id=_document_id(source, digest),
        digest=digest,
        automatic=automatic,
        provenance=provenance,
        warnings=warnings,
    )
    record.update(
        {
            "physical_pages": int(total_pages),
            "extracted_units": 0,
            "empty_units": 0,
            **empty_removal_counts(),
        }
    )
    record["extraction_backend"] = docling_backend_record_from_env(docling_env)
    return record


def _docling_page_units(
    raw_units: list[dict[str, Any]],
    document: dict[str, Any],
    page_number: int,
    page_label: str,
) -> list[dict[str, Any]]:
    locator = {
        "type": "pdf_page",
        "page": page_number,
        "page_label": page_label,
    }
    units: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_units, 1):
        text = normalize_reading_text(str(raw.get("contents") or ""))
        if not text.strip():
            continue
        kind = str(raw.get("content_kind") or "prose")
        unit = {
            "id": (
                f"{document['document_id']}:pdf-page:{page_number:06d}:"
                f"docling:{index:04d}"
            ),
            "document_id": str(document["document_id"]),
            "source_id": str(document["source_id"]),
            "title": str(document["title"]),
            "contents": text,
            "content_kind": kind,
            "annotations": [],
            "quality_flags": _quality_flags(text, kind),
            "locator": locator,
            "provenance": list(raw.get("provenance") or []),
        }
        # Structural labels the adapter attached to a raw unit travel with it, so
        # the shared gate can read them. A raw unit without them keeps no key.
        for field in ("docling_label", "docling_content_layer", "docling_ref"):
            if (value := raw.get(field)) is not None:
                unit[field] = value
        units.append(unit)
    return units


def _docling_incompleteness(diagnostics: dict[str, Any]) -> str | None:
    """Why a conversion could have lost evidence, or None when it is complete.

    An unsupported item was acknowledged but not emitted, and an unresolved
    child or caption reference means a picture's descendants could not be
    classified. Either can lose evidence, so the source is refused before it can
    reach the success path; a clean conversion keeps its diagnostics.
    """

    unsupported = int(diagnostics.get("unsupported_item_count") or 0)
    missing = int(diagnostics.get("missing_child_ref_count") or 0) + int(
        diagnostics.get("missing_caption_ref_count") or 0
    )
    if not unsupported and not missing:
        return None
    return f"unsupported_item_count={unsupported}, unresolved_reference_count={missing}"


def _annotate_docling_batch(
    result: dict[str, Any],
    document: dict[str, Any],
    *,
    page_first: int,
    page_last: int,
) -> tuple[
    dict[int, list[dict[str, Any]]],
    int,
    dict[str, int],
    dict[str, int],
]:
    labels = {
        int(key): str(value) for key, value in (result.get("page_labels") or {}).items()
    }
    pages: dict[int, list[dict[str, Any]]] = {}
    populated: set[int] = set()
    for page_key, raw_units in (result.get("pages") or {}).items():
        page_number = int(page_key)
        units = _docling_page_units(
            list(raw_units or []),
            document,
            page_number,
            labels.get(page_number) or str(page_number),
        )
        if units:
            pages[page_number - 1] = units
            populated.add(page_number)
    # A blank page is one the worker returned nothing for, so it carries no page
    # key at all. Count every requested page that produced no unit, once; an
    # image-only picture page is such a page and is not counted again anywhere.
    empty = sum(
        1
        for page_number in range(page_first, page_last + 1)
        if page_number not in populated
    )
    diagnostics = {
        str(key): int(value)
        for key, value in (result.get("diagnostics") or {}).items()
        if isinstance(value, (int, float))
    }
    removals = empty_removal_counts()
    removals["removed_repeated_margin_blocks"] = int(
        diagnostics.get("excluded_furniture_item_count") or 0
    )
    # Image-only pages are a source-wide diagnostic from the native probe, not a
    # second count of the empty pages above.
    removals["image_only_pages"] = int(result.get("image_only_pages") or 0)
    return pages, empty, removals, diagnostics


def docling_pdf_batch(
    source: SourceFile,
    existing_document: dict[str, Any] | None,
    start_index: int,
    end_index: int,
    *,
    digest: str,
    offline: bool,
    model_cache_root: Path | None,
    docling_env: DoclingEnv,
    total_pages: int | None = None,
    worker: DoclingWorker | None = None,
) -> tuple[
    dict[int, list[dict[str, Any]]],
    dict[str, Any],
    int,
    dict[str, int],
    dict[str, int],
]:
    """One bounded Docling page batch for the staged build path.

    ``start_index`` and ``end_index`` are 0-based and half-open. The document
    record is built on the first batch and passed back unchanged afterwards, so
    every batch shares one identity. ``docling_env`` is the managed environment
    resolved once at the start of the build, and ``worker`` the build's warm
    worker for it. Raises :class:`ExtractionError` for a source-local conversion
    failure, which the staged caller omits like any other unreadable source.
    """

    try:
        result = run_docling_batch(
            source_path=source.path,
            start=start_index + 1,
            end=end_index,
            offline=offline,
            model_cache_root=model_cache_root,
            docling_env=docling_env,
            worker=worker,
        )
    except DoclingConversionError as exc:
        raise ExtractionError(str(exc)) from exc
    incompleteness = _docling_incompleteness(result.get("diagnostics") or {})
    if incompleteness is not None:
        raise ExtractionError(
            "Docling could not convert "
            f"{source.source_relative_path} pages {start_index + 1}-{end_index} "
            f"without risking evidence loss ({incompleteness}); the source is "
            "omitted from this generation rather than partially trusted."
        )
    if existing_document is None:
        document = _docling_document_record(
            source,
            digest,
            result.get("pdf_metadata") or {},
            int(total_pages) if total_pages else end_index,
            docling_env,
        )
    else:
        document = existing_document
    pages, empty, removals, diagnostics = _annotate_docling_batch(
        result,
        document,
        page_first=start_index + 1,
        page_last=end_index,
    )
    return pages, document, empty, removals, diagnostics
