"""Layout-aware extraction and bibliographic identity for PDF/EPUB sources."""

from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from statistics import median
from typing import Any

import ebooklib
import pymupdf
from bs4 import BeautifulSoup, Comment, NavigableString, Tag
from ebooklib import epub

from .settings import BM25_STOPWORD_LANGUAGES, bm25_stopwords
from .sources import SourceFile, sha256_file

pymupdf.no_recommend_layout()


class ExtractionError(RuntimeError):
    """Raised when a source cannot be represented safely in the knowledge base."""


_HORIZONTAL_SPACE = re.compile(r"[\t\f\v \u00a0]+")
_LIST_ITEM = re.compile(r"^(?:[-*•]|\d+[.)]|[A-Za-z][.)])\s+")
_CONTROL_CHARACTERS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
# Compatibility folding is deliberately limited to the two blocks that English
# scholarship actually produces: Mathematical Alphanumeric Symbols (letters that
# come from formula fonts and are otherwise unmatchable by typed queries) and
# Alphabetic Presentation Forms (fi/fl/ff ligatures). Global NFKC would also fold
# superscripts, subscripts, and symbols that carry meaning in citations.
_FOLDABLE_CHARACTERS = re.compile(r"[\U0001d400-\U0001d7ff\ufb00-\ufb06\ufb13-\ufb17]")
_DOI = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.IGNORECASE)
_URL = re.compile(r"^(?:https?://|www\.|(?:dx\.)?doi\.org/)", re.IGNORECASE)
_PAGE_NUMBER = re.compile(
    r"^(?:page\s+)?(?:[ivxlcdm]+|\d+)(?:\s+(?:of|/|-|–)\s*\d+)?$",
    re.IGNORECASE,
)
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
_AUTHOR_EXCLUSIONS = re.compile(
    r"^(?:abstract|keywords?|volume|number|doi|issn|copyright|published|received|"
    r"accepted|introduction)\b",
    re.IGNORECASE,
)
_YEAR = re.compile(r"\b(18\d{2}|19\d{2}|20\d{2}|21\d{2})\b")
_PDF_DATE_YEAR = re.compile(r"^(?:D:)?(18\d{2}|19\d{2}|20\d{2}|21\d{2})")
_DATED_LINE = re.compile(
    r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|"
    r"dec(?:ember)?)\b.*\b(?:18\d{2}|19\d{2}|20\d{2}|21\d{2})\b",
    re.IGNORECASE,
)
_LEADING_ASTERISK_MARKER = re.compile(r"^\s*\*\s*")
_TRAILING_ASTERISK_MARKER = re.compile(r"(?<=\w)\*\s*$")
_EXPLICIT_MARKER_LEGEND = re.compile(
    r"^\s*\*\s*(?:(?:indicates?|denotes?|means?)\b|[:=])",
    re.IGNORECASE,
)
_BYLINE = re.compile(r"^(?:by|written\s+by|author(?:s)?\s*:)\s+", re.IGNORECASE)
_AUTHOR_JUNK = re.compile(
    r"^(?:unknown|anonymous|untitled|microsoft\s+word|adobe\s+(?:acrobat|indesign)|"
    r"administrator|admin|user)$",
    re.IGNORECASE,
)
_TITLE_LABEL = re.compile(
    r"^(?:review\s+of\s+the\s+month|research\s+article|original\s+article|article)$",
    re.IGNORECASE,
)
_AUTHOR_AFFILIATION = re.compile(
    r"\s*\([^)]*(?:university|institute|college)[^)]*\)\s*$", re.IGNORECASE
)
_AUTHOR_MARKER = re.compile(r"(?:\d+(?:\s*,\s*\d+)*|[*∗†‡§])+$")
_FIGURE_CAPTION = re.compile(r"^(?:fig(?:ure)?\.?)\s*(?:\d|[ivxlcdm])", re.IGNORECASE)
_TABLE_CAPTION = re.compile(r"^table\s*(?:\d|[ivxlcdm])", re.IGNORECASE)
_MOJIBAKE_MARKERS = ("â€", "ï¿½", "ðŸ")
_MOJIBAKE_LATIN1_PAIR = re.compile(r"(?:Ã|Â)[\u0080-\u00bf]")
_EPUB_HEADING_ELEMENTS = frozenset({"h1", "h2", "h3", "h4", "h5", "h6"})
_EPUB_CONTENT_ELEMENTS = frozenset({"p", "li", "blockquote", "table"})
_EPUB_SEMANTIC_ELEMENTS = _EPUB_HEADING_ELEMENTS | _EPUB_CONTENT_ELEMENTS


@dataclass(frozen=True, slots=True)
class _TextBlock:
    number: int
    bbox: tuple[float, float, float, float]
    lines: tuple[str, ...]
    text: str
    font_size: float

    @property
    def width(self) -> float:
        return max(0.0, self.bbox[2] - self.bbox[0])


@dataclass(frozen=True, slots=True)
class _Region:
    kind: str
    bbox: tuple[float, float, float, float]


def _starts_with_alpha(value: str) -> bool:
    for character in value:
        if character.isalpha():
            return True
        if character.isdigit():
            return False
    return False


def _ends_sentence(value: str) -> bool:
    return value.rstrip("\"'”’)]}").endswith((".", "!", "?", "…", ":"))


def _fold_compatibility_characters(value: str) -> str:
    """Fold formula-font letters and presentation ligatures to plain text."""

    if not _FOLDABLE_CHARACTERS.search(value):
        return value
    return _FOLDABLE_CHARACTERS.sub(
        lambda match: unicodedata.normalize("NFKC", match.group(0)),
        value,
    )


def normalize_reading_text(value: str) -> str:
    """Remove extraction layout wrapping without rewriting source prose."""

    text = unicodedata.normalize("NFC", _fold_compatibility_characters(value))
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\u00ad", "")
    text = _CONTROL_CHARACTERS.sub("", text)
    paragraphs: list[str] = []
    current = ""
    separated = False
    for raw_line in text.splitlines():
        line = _HORIZONTAL_SPACE.sub(" ", raw_line).strip()
        # PDF text spans sometimes leave a layout-only space after a hyphen
        # even though the printed form is a normal hyphenated word.
        line = re.sub(r"(?<=[^\W\d_])-\s+(?=[^\W\d_])", "-", line)
        if not line:
            separated = bool(current)
            continue
        if not current:
            current = line
        elif _LIST_ITEM.match(line) or (separated and _ends_sentence(current)):
            paragraphs.append(current)
            current = line
        elif (
            current.endswith("-")
            and current[-2:-1].isalpha()
            and _starts_with_alpha(line)
        ):
            current = current[:-1] + line
        else:
            current = f"{current} {line}"
        separated = False
    if current:
        paragraphs.append(current)
    return "\n\n".join(paragraphs)


def normalize_inline_text(value: str) -> str:
    """Normalize extracted metadata to one display-safe line."""

    return _HORIZONTAL_SPACE.sub(
        " ", normalize_reading_text(value).replace("\n", " ")
    ).strip()


def _script_family(character: str) -> str | None:
    """Return a stable, dependency-free script family for alphabetic text."""

    if not character.isalpha():
        return None
    name = unicodedata.name(character, "")
    for prefix, family in (
        ("LATIN ", "latin"),
        ("CJK ", "cjk"),
        ("IDEOGRAPHIC ", "cjk"),
        ("HIRAGANA ", "japanese"),
        ("KATAKANA ", "japanese"),
        ("HANGUL ", "hangul"),
        ("CYRILLIC ", "cyrillic"),
        ("GREEK ", "greek"),
        ("ARABIC ", "arabic"),
        ("HEBREW ", "hebrew"),
        ("ARMENIAN ", "armenian"),
        ("DEVANAGARI ", "devanagari"),
        ("BENGALI ", "bengali"),
        ("GURMUKHI ", "gurmukhi"),
        ("GUJARATI ", "gujarati"),
        ("ORIYA ", "oriya"),
        ("TAMIL ", "tamil"),
        ("TELUGU ", "telugu"),
        ("KANNADA ", "kannada"),
        ("MALAYALAM ", "malayalam"),
        ("SINHALA ", "sinhala"),
        ("THAI ", "thai"),
        ("LAO ", "lao"),
        ("TIBETAN ", "tibetan"),
        ("MYANMAR ", "myanmar"),
        ("GEORGIAN ", "georgian"),
        ("ETHIOPIC ", "ethiopic"),
        ("CHEROKEE ", "cherokee"),
        ("CANADIAN SYLLABICS ", "canadian_syllabics"),
        ("MONGOLIAN ", "mongolian"),
        ("THAANA ", "thaana"),
        ("COPTIC ", "coptic"),
    ):
        if name.startswith(prefix):
            return family
    return name.split(" ", 1)[0].casefold() if name else "unknown"


def _text_signals(value: str) -> tuple[list[str], list[str]]:
    """Return corruption evidence and advisory script notes for one text value.

    Corruption evidence withholds text from retrieval. Script notes never do:
    English-language scholarship legitimately quotes Greek, Cyrillic, Arabic, and
    other scripts, so a mixed-script passage is reported for inspection instead of
    being discarded.
    """

    raw = unicodedata.normalize("NFC", value)
    normalized = normalize_inline_text(raw)
    if not normalized:
        return [], []
    replacement_count = normalized.count("\ufffd")
    private_or_unassigned = sum(
        unicodedata.category(character) in {"Co", "Cn", "Cs"}
        for character in normalized
    )
    alphabetic = [character for character in normalized if character.isalpha()]
    families = Counter(
        family
        for character in alphabetic
        if (family := _script_family(character)) is not None
    )
    latin_count = families.get("latin", 0)
    dominant_count = max(families.values(), default=0)

    corruption: list[str] = []
    mojibake = bool(
        _MOJIBAKE_LATIN1_PAIR.search(raw)
        or any(marker in raw for marker in _MOJIBAKE_MARKERS)
    )
    # One replacement character is only evidence of corruption when another
    # corruption signal corroborates it. Script mixing must not corroborate,
    # because that is how a legitimate foreign-language quotation was withheld.
    corroborated = bool(private_or_unassigned or mojibake)
    if replacement_count >= 2 or (replacement_count == 1 and corroborated):
        corruption.append("replacement_characters")
    if private_or_unassigned >= 2 or (
        private_or_unassigned == 1 and replacement_count > 0
    ):
        corruption.append("private_or_unassigned_characters")
    if mojibake:
        corruption.append("known_mojibake")

    notes: list[str] = []
    if len(alphabetic) >= 20 and latin_count / len(alphabetic) < 0.50:
        notes.append("non_latin_dominant")
    if len(families) >= 4 and dominant_count / len(alphabetic) < 0.70:
        notes.append("mixed_script_text")
    return corruption, notes


def text_corruption_reasons(value: str) -> list[str]:
    """Return the corruption evidence that withholds extraction text.

    Only incoherent output is withheld: replacement characters, private-use or
    unassigned code points, and known damaged encoding sequences. Script mixing
    and non-Latin dominance are reported by `text_script_notes` instead, so
    legitimate quotations remain retrievable.
    """

    return _text_signals(value)[0]


def text_script_notes(value: str) -> list[str]:
    """Return advisory non-Latin or mixed-script notes that never withhold text."""

    return _text_signals(value)[1]


def has_searchable_alphanumeric_content(value: str) -> bool:
    """Return whether normalized text contains a Unicode letter or number."""

    return any(character.isalnum() for character in normalize_inline_text(value))


def text_health_reasons(value: str) -> list[str]:
    """Identify extraction text that is corrupt or has no searchable content."""

    reasons = text_corruption_reasons(value)
    normalized = normalize_inline_text(value)
    if normalized and not has_searchable_alphanumeric_content(normalized):
        reasons.append("symbol_only")
    return reasons


# Retrieval rejects a candidate for exactly two reasons, and both are properties
# of the chunk text plus its stored quality flags rather than of the query. They
# are therefore computed once when the artifact lookup is built and stored as a
# bitmask, which removes the per-query text scans from the candidate gate.
CHUNK_FLAG_CORRUPT_TEXT = 1
CHUNK_FLAG_EXTRACTION_ARTIFACT = 2


def chunk_health_flags(text: str, *, quality_flags: object = None) -> int:
    """Return the precomputed retrieval-rejection verdict for one chunk.

    A chunk is an extraction artifact when it carries that quality flag or when
    it has no searchable alphanumeric content, which mirrors the query-time
    check exactly so the rejection counters cannot change.
    """

    flags = 0
    if text_corruption_reasons(text):
        flags |= CHUNK_FLAG_CORRUPT_TEXT
    stored = {str(item) for item in quality_flags or ()}
    if "extraction_artifact" in stored or not has_searchable_alphanumeric_content(text):
        flags |= CHUNK_FLAG_EXTRACTION_ARTIFACT
    return flags


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
    comparable_title = re.sub(r"\.(?:pdf|epub)$", "", normalized, flags=re.IGNORECASE)
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
    if _ends_sentence(cleaned) or len(cleaned.split()) > 32:
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


def _title_and_embedded_byline(
    blocks: list[_TextBlock],
) -> tuple[str, list[str]]:
    lines = [line for block in blocks for line in block.lines]
    authors: list[str] = []
    if len(lines) >= 3 and lines[1].rstrip().endswith(":"):
        possible_author = _author_names(lines[0])
        if possible_author:
            authors = possible_author
            lines = lines[1:]

    cleaned_lines: list[str] = []
    for line in lines:
        normalized = normalize_inline_text(line)
        if (
            not normalized
            or _TITLE_LABEL.fullmatch(normalized)
            or _FRONT_MATTER_EXCLUSIONS.match(normalized)
            or _URL.match(normalized)
        ):
            continue
        cleaned_lines.append(normalized)
    # Preserve the original line boundaries long enough for the common
    # ``inter-\nword`` PDF wrapping case to be joined as ``interword``.
    title = normalize_inline_text(normalize_reading_text("\n".join(cleaned_lines)))
    title = re.sub(
        r"\s+arxiv:\d{4}\.\d+(?:v\d+)?(?:\s+\[[^]]+\])?(?:\s+\d{1,2}\s+\w+\s+\d{4})?$",
        "",
        title,
        flags=re.IGNORECASE,
    ).strip()
    return title, authors


# A language is detected from how much of a language's own function-word
# inventory a text sample uses. Coverage rather than raw share, because bm25s's
# lists are wildly different sizes (33 English words against 499 Korean ones) and
# a share would hand every sample to the longest list.
_LANGUAGE_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)
_LANGUAGE_SAMPLE_TOKENS = 4000
_LANGUAGE_MINIMUM_TOKENS = 120
_LANGUAGE_MINIMUM_COVERAGE = 0.10
_LANGUAGE_MINIMUM_MARGIN = 1.5
_LANGUAGE_TAG = re.compile(r"^([A-Za-z]{2,3})(?:[-_].*)?$")


def _iso_language_code(value: str) -> str:
    """Reduce a declared language tag such as `en-US` or `eng` to a code."""

    match = _LANGUAGE_TAG.match(str(value).strip())
    return match.group(1).casefold() if match else ""


def _detect_language(text: str) -> str:
    """Return the code whose stopword list a text sample covers most fully.

    The candidate lists are the ones BM25 filters with, so a language detected
    here is one whose function words the lexical half will ignore. A sample with
    too few tokens, one whose best candidate does not clear a coverage floor, or
    one whose best candidate does not clearly beat the runner-up returns empty
    rather than a guess, and the caller records the language as missing. The floor
    is what rejects a language outside the candidate set, whose texts cover no
    list at all. CJK text is not separated into words here, so those sources
    report no detection.
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


def _pdf_line_text(spans: list[dict[str, Any]]) -> str:
    """Join extracted spans without guessing replacements for ambiguous glyphs."""

    return "".join(str(span.get("text") or "") for span in spans)


def _page_blocks(page: pymupdf.Page) -> list[_TextBlock]:
    result: list[_TextBlock] = []
    payload = page.get_text("dict", sort=False, flags=pymupdf.TEXTFLAGS_TEXT)
    for number, block in enumerate(payload.get("blocks", [])):
        if block.get("type") != 0:
            continue
        lines: list[str] = []
        sizes: list[float] = []
        for line in block.get("lines", []):
            value = _pdf_line_text(line.get("spans", []))
            value = _HORIZONTAL_SPACE.sub(" ", value).strip()
            if value:
                lines.append(value)
            sizes.extend(
                float(span.get("size") or 0.0)
                for span in line.get("spans", [])
                if span.get("text")
            )
        if not lines:
            continue
        bbox = tuple(float(item) for item in block.get("bbox", (0, 0, 0, 0)))
        result.append(
            _TextBlock(
                number=number,
                bbox=(bbox[0], bbox[1], bbox[2], bbox[3]),
                lines=tuple(lines),
                text="\n".join(lines),
                font_size=max(sizes, default=0.0),
            )
        )
    return result


def _front_matter_identity(
    pages: list[tuple[list[_TextBlock], float, float]],
    metadata: dict[str, Any],
    filename_stem: str,
    declared_language: str = "",
) -> tuple[dict[str, Any], dict[str, str], list[str]]:
    embedded_title = normalize_inline_text(str(metadata.get("title") or ""))
    embedded_authors = _pdf_author_list(metadata)
    visible_title = ""
    visible_authors: list[str] = []
    text_bearing_pages = [page for page in pages if page[0]][:5]
    declared_code = _iso_language_code(declared_language)
    language = declared_code or _detect_language(
        "\n".join(
            block.text
            for blocks, _width, _height in text_bearing_pages
            for block in blocks
        )
    )
    language_source = (
        "pdf_catalog" if declared_code else "text_sample" if language else "missing"
    )
    first_page_text = (
        "\n".join(block.text for block in text_bearing_pages[0][0])
        if text_bearing_pages
        else ""
    )
    # A DOI in an arbitrary prose sentence is often a cited work, not this
    # document. Visible DOI candidates therefore need an explicit DOI cue.
    visible_doi_text = "\n".join(
        line
        for line in first_page_text.splitlines()
        if re.search(r"(?:\bdoi\b|https?://doi\.org/)", line, re.IGNORECASE)
    )
    publication_text = "\n".join(
        block.text
        for blocks, _width, _height in text_bearing_pages[:2]
        for block in blocks
    )

    for blocks, _width, height in text_bearing_pages:
        if not blocks:
            continue
        vertical_limit = height * (0.95 if len(blocks) <= 5 else 0.62)
        candidates = [
            block
            for block in blocks
            if block.bbox[1] <= vertical_limit
            and not _FRONT_MATTER_EXCLUSIONS.match(normalize_inline_text(block.text))
            and _valid_title(
                _title_and_embedded_byline([block])[0],
                filename_stem,
                reject_filename_match=False,
            )
        ]
        body_sizes = [
            block.font_size
            for block in blocks
            if len(normalize_inline_text(block.text)) >= 200 and block.font_size > 0
        ]
        all_sizes = [block.font_size for block in blocks if block.font_size > 0]
        body_size = median(body_sizes or [min(all_sizes, default=10.0)])
        if not candidates:
            continue
        maximum = max(block.font_size for block in candidates)
        minimum_title_size = 12.0 if len(blocks) <= 5 else max(12.0, body_size * 1.22)
        if maximum < minimum_title_size:
            continue
        anchor = max(
            (block for block in candidates if block.font_size >= maximum * 0.98),
            key=lambda block: (
                len(_title_and_embedded_byline([block])[0]),
                -block.bbox[1],
            ),
        )
        top = anchor.bbox[1]
        title_blocks = sorted(
            [
                block
                for block in candidates
                if block.font_size >= maximum * 0.79
                and top - 20 <= block.bbox[1] <= top + 110
            ],
            key=lambda item: (item.bbox[1], item.bbox[0]),
        )
        if anchor not in title_blocks:
            title_blocks.append(anchor)
            title_blocks.sort(key=lambda item: (item.bbox[1], item.bbox[0]))
        visible_title, embedded_byline = _title_and_embedded_byline(title_blocks)
        title_bottom = max(block.bbox[3] for block in title_blocks)
        visible_authors = embedded_byline
        if not visible_authors:
            possible_authors = sorted(
                [
                    block
                    for block in blocks
                    if title_bottom < block.bbox[1] <= title_bottom + 360
                    and len(normalize_inline_text(block.text)) <= 240
                    and not _AUTHOR_EXCLUSIONS.match(normalize_inline_text(block.text))
                ],
                key=lambda item: (item.bbox[1], item.bbox[0]),
            )
            explicit_bylines = [
                block
                for block in possible_authors
                if _BYLINE.match(normalize_inline_text(block.text))
            ]
            for author_block in explicit_bylines or possible_authors:
                if author_names := _author_names(author_block.text):
                    visible_authors = author_names
                    break
        break

    warnings: list[str] = []
    if visible_title:
        title = visible_title
        title_source = "pdf_front_matter"
        if _valid_title(embedded_title, filename_stem) and (
            normalize_inline_text(embedded_title).casefold() != visible_title.casefold()
        ):
            warnings.append("conflicting_candidates")
    elif _valid_title(embedded_title, filename_stem):
        title = embedded_title
        title_source = "pdf_metadata"
    else:
        title = filename_stem
        title_source = "filename"

    if visible_authors:
        authors = visible_authors
        authors_source = "pdf_front_matter"
    elif embedded_authors:
        authors = embedded_authors
        authors_source = "pdf_metadata"
    else:
        authors = []
        authors_source = "missing"

    doi = ""
    doi_source = "missing"
    for candidate, source_name in (
        (embedded_title, "pdf_metadata"),
        (" ".join(str(value) for value in metadata.values()), "pdf_metadata"),
        (visible_doi_text, "pdf_front_matter"),
    ):
        if extracted := _doi(candidate):
            doi = extracted
            doi_source = source_name
            break

    year: int | None = None
    year_source = "missing"
    for line in publication_text.splitlines():
        normalized = normalize_inline_text(line)
        has_publication_cue = bool(
            re.search(
                r"(?:©|copyright|published|volume|number|issue)",
                normalized,
                re.IGNORECASE,
            )
            or (len(normalized) <= 100 and _DATED_LINE.search(normalized))
        )
        if has_publication_cue and (match := _YEAR.search(normalized)):
            year = int(match.group(1))
            year_source = "pdf_front_matter"
            break
    if year is None:
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
            "authors": authors_source,
            "year": year_source,
            "doi": doi_source,
            "language": language_source,
        },
        warnings,
    )


def _pdf_declared_language(document: pymupdf.Document) -> str:
    """Return the language a PDF declares for itself, or an empty string.

    A PDF may carry a `/Lang` entry in its catalog, which PyMuPDF exposes as
    `Document.language`. Many carry none, and the caller falls back to detection
    over the text sample.
    """

    try:
        return str(document.language or "")
    except (RuntimeError, ValueError):
        return ""


def _block_signature(block: _TextBlock) -> str:
    value = normalize_inline_text(block.text).casefold()
    return re.sub(r"\d+", "#", value)


def _repeated_margin_signatures(
    pages: list[tuple[list[_TextBlock], float, float]],
) -> set[str]:
    if len(pages) < 3:
        return set()
    page_sets: list[set[str]] = []
    for blocks, _width, height in pages:
        page_sets.append(
            {
                signature
                for block in blocks
                if (block.bbox[3] <= height * 0.12 or block.bbox[1] >= height * 0.88)
                and (signature := _block_signature(block))
            }
        )
    counts = Counter(signature for values in page_sets for signature in values)
    threshold = max(3, math.ceil(len(pages) * 0.30))
    return {signature for signature, count in counts.items() if count >= threshold}


def _rect_area(rect: tuple[float, float, float, float]) -> float:
    return max(0.0, rect[2] - rect[0]) * max(0.0, rect[3] - rect[1])


def _rect_intersection_ratio(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    intersection = (
        max(left[0], right[0]),
        max(left[1], right[1]),
        min(left[2], right[2]),
        min(left[3], right[3]),
    )
    return _rect_area(intersection) / max(_rect_area(left), 1.0)


def _merge_rectangles(
    rectangles: list[pymupdf.Rect], padding: float = 8.0
) -> list[pymupdf.Rect]:
    merged: list[pymupdf.Rect] = []
    for original in rectangles:
        candidate = pymupdf.Rect(original)
        changed = True
        while changed:
            changed = False
            remaining: list[pymupdf.Rect] = []
            padded = candidate + (-padding, -padding, padding, padding)
            for existing in merged:
                if padded.intersects(existing):
                    candidate |= existing
                    changed = True
                else:
                    remaining.append(existing)
            merged = remaining
        merged.append(candidate)
    return merged


def _non_prose_regions(page: pymupdf.Page) -> list[_Region]:
    regions: list[_Region] = []
    try:
        finder = page.find_tables()
    except (RuntimeError, ValueError):
        finder = None
    if finder is not None:
        for table in finder.tables:
            bbox = tuple(float(item) for item in table.bbox)
            regions.append(_Region("table", bbox))

    payload = page.get_text("dict", sort=False, flags=pymupdf.TEXTFLAGS_TEXT)
    for block in payload.get("blocks", []):
        if block.get("type") == 1:
            bbox = tuple(float(item) for item in block.get("bbox", (0, 0, 0, 0)))
            if _rect_area(bbox) >= page.rect.get_area() * 0.02:
                regions.append(_Region("figure", bbox))

    try:
        drawing_rectangles = [
            pymupdf.Rect(item["rect"])
            for item in page.get_drawings()
            if item.get("rect") is not None
            and pymupdf.Rect(item["rect"]).get_area() > 4
        ]
    except (RuntimeError, ValueError):
        drawing_rectangles = []
    for rect in _merge_rectangles(drawing_rectangles):
        if rect.get_area() >= page.rect.get_area() * 0.025:
            expanded = rect + (-12, -24, 12, 12)
            expanded &= page.rect
            regions.append(_Region("figure", tuple(float(item) for item in expanded)))

    figures = [region for region in regions if region.kind == "figure"]
    regions = [
        region
        for region in regions
        if region.kind != "table"
        or not any(
            _rect_intersection_ratio(region.bbox, figure.bbox) >= 0.55
            for figure in figures
        )
    ]
    deduplicated: list[_Region] = []
    for region in sorted(
        regions, key=lambda item: (item.kind != "table", -_rect_area(item.bbox))
    ):
        if any(
            _rect_intersection_ratio(region.bbox, item.bbox) > 0.8
            for item in deduplicated
        ):
            continue
        deduplicated.append(region)
    return sorted(deduplicated, key=lambda item: (item.bbox[1], item.bbox[0]))


def _sort_band(blocks: list[_TextBlock], page_width: float) -> list[_TextBlock]:
    center = page_width / 2
    left = [block for block in blocks if block.bbox[2] <= center * 1.03]
    right = [block for block in blocks if block.bbox[0] >= center * 0.97]
    other = [block for block in blocks if block not in left and block not in right]
    if len(left) >= 2 and len(right) >= 2 and not other:
        return sorted(left, key=lambda item: (item.bbox[1], item.bbox[0])) + sorted(
            right, key=lambda item: (item.bbox[1], item.bbox[0])
        )
    return sorted(blocks, key=lambda item: (item.bbox[1], item.bbox[0]))


def _reading_order(blocks: list[_TextBlock], page_width: float) -> list[_TextBlock]:
    full_width = [block for block in blocks if block.width >= page_width * 0.62]
    narrow = [block for block in blocks if block not in full_width]
    ordered: list[_TextBlock] = []
    previous_bottom = -1.0
    for anchor in sorted(full_width, key=lambda item: (item.bbox[1], item.bbox[0])):
        band = [
            block
            for block in narrow
            if previous_bottom <= block.bbox[1] < anchor.bbox[1]
        ]
        ordered.extend(_sort_band(band, page_width))
        ordered.append(anchor)
        previous_bottom = anchor.bbox[3]
    ordered.extend(
        _sort_band([block for block in narrow if block not in ordered], page_width)
    )
    return list(dict.fromkeys(ordered))


def _marker_annotations(blocks: list[_TextBlock]) -> list[dict[str, Any]]:
    legend = ""
    for block in blocks:
        normalized = normalize_inline_text(block.text)
        if _EXPLICIT_MARKER_LEGEND.search(normalized):
            legend = normalized
        if legend:
            break
    if not legend:
        return []
    affected: list[str] = []
    for block in blocks:
        for line in block.lines:
            if (
                line.count("*") != 1
                or not _TRAILING_ASTERISK_MARKER.search(line)
                or _EXPLICIT_MARKER_LEGEND.search(line)
            ):
                continue
            cleaned = normalize_inline_text(
                _TRAILING_ASTERISK_MARKER.sub("", line, count=1)
            )
            if cleaned and cleaned not in affected:
                affected.append(cleaned)
    return [
        {
            "type": "legend_marker",
            "marker": "*",
            "legend_text": legend,
            "meaning": normalize_inline_text(
                _LEADING_ASTERISK_MARKER.sub("", legend, count=1)
            ),
            "applies_to": affected,
        }
    ]


def _vertical_gap(
    block: tuple[float, float, float, float],
    region: tuple[float, float, float, float],
) -> float:
    if block[3] < region[1]:
        return region[1] - block[3]
    if block[1] > region[3]:
        return block[1] - region[3]
    return 0.0


def _horizontal_overlap_ratio(
    block: tuple[float, float, float, float],
    region: tuple[float, float, float, float],
) -> float:
    overlap = max(0.0, min(block[2], region[2]) - max(block[0], region[0]))
    return overlap / max(1.0, min(block[2] - block[0], region[2] - region[0]))


def _nearby_region_labels(
    region: _Region,
    blocks: list[_TextBlock],
) -> list[_TextBlock]:
    """Attach only explicit, nearby captions and marker legends to a region."""

    caption_pattern = _TABLE_CAPTION if region.kind == "table" else _FIGURE_CAPTION
    result: list[_TextBlock] = []
    for block in blocks:
        normalized = normalize_inline_text(block.text)
        explicit_caption = bool(caption_pattern.match(normalized))
        explicit_legend = bool(_EXPLICIT_MARKER_LEGEND.search(normalized))
        if not (explicit_caption or explicit_legend):
            continue
        if _vertical_gap(block.bbox, region.bbox) > 96:
            continue
        if _horizontal_overlap_ratio(block.bbox, region.bbox) < 0.20:
            continue
        result.append(block)
    return sorted(result, key=lambda item: (item.bbox[1], item.bbox[0]))


def _label_annotations(
    kind: str,
    blocks: list[_TextBlock],
) -> list[dict[str, Any]]:
    caption_pattern = _TABLE_CAPTION if kind == "table" else _FIGURE_CAPTION
    return [
        {
            "type": "caption",
            "content_kind": kind,
            "text": normalize_reading_text(block.text),
        }
        for block in blocks
        if caption_pattern.match(normalize_inline_text(block.text))
    ]


def _quality_flags(text: str, kind: str) -> list[str]:
    """Mark extraction debris that meets deterministic rejection rules."""

    normalized = normalize_inline_text(text)
    alphabetic = sum(character.isalpha() for character in normalized)
    flags: list[str] = []
    if not has_searchable_alphanumeric_content(normalized) or (
        kind == "prose" and alphabetic < 3
    ):
        flags.append("extraction_artifact")
    return flags


def _split_prose_and_lists(
    blocks: list[_TextBlock],
) -> list[tuple[str, list[_TextBlock]]]:
    """Preserve reading order while separating list blocks from prose blocks."""

    groups: list[tuple[str, list[_TextBlock]]] = []
    for block in blocks:
        populated_lines = [line for line in block.lines if line.strip()]
        list_lines = sum(
            bool(_LIST_ITEM.match(line.strip())) for line in populated_lines
        )
        kind = (
            "list"
            if populated_lines and list_lines * 2 >= len(populated_lines)
            else "prose"
        )
        if groups and groups[-1][0] == kind:
            groups[-1][1].append(block)
        else:
            groups.append((kind, [block]))
    return groups


def _pdf_locator(page: pymupdf.Page, page_number: int) -> dict[str, Any]:
    # PyMuPDF's own label lookup filters the document's label tree and indexes
    # the result, so a tree that starts after the page being asked about makes it
    # index an empty list and raise IndexError: a document whose labels begin on
    # page 2 raises on page 1. A locator has to resolve to something, and the
    # physical page number always does.
    try:
        page_label = str(page.get_label() or page_number)
    except (RuntimeError, ValueError, IndexError):
        page_label = str(page_number)
    return {"type": "pdf_page", "page": page_number, "page_label": page_label}


def _pdf_page_units(
    page: pymupdf.Page,
    *,
    page_number: int,
    blocks: list[_TextBlock],
    page_width: float,
    page_height: float,
    repeated_margins: set[str],
    document_id: str,
    source_id: str,
    title: str,
) -> tuple[list[dict[str, Any]], bool, int]:
    locator = _pdf_locator(page, page_number)
    filtered: list[_TextBlock] = []
    removed_margin_blocks = 0
    for block in blocks:
        in_margin = (
            block.bbox[3] <= page_height * 0.12 or block.bbox[1] >= page_height * 0.88
        )
        if in_margin and (
            _block_signature(block) in repeated_margins
            or _PAGE_NUMBER.fullmatch(normalize_inline_text(block.text))
        ):
            removed_margin_blocks += 1
            continue
        filtered.append(block)
    if not filtered:
        return [], True, removed_margin_blocks

    regions = _non_prose_regions(page)
    assigned: set[int] = set()
    page_units: list[tuple[str, list[_TextBlock]]] = []
    for region in regions:
        members = [
            block
            for block in filtered
            if _rect_intersection_ratio(block.bbox, region.bbox) >= 0.35
        ]
        nearby_labels = _nearby_region_labels(
            region,
            [
                block
                for block in filtered
                if block.number not in assigned and block not in members
            ],
        )
        members.extend(nearby_labels)
        if not members:
            continue
        assigned.update(block.number for block in members)
        page_units.append((region.kind, members))
    prose = [block for block in filtered if block.number not in assigned]
    if prose:
        page_units[0:0] = _split_prose_and_lists(_reading_order(prose, page_width))

    units: list[dict[str, Any]] = []
    for region_index, (kind, members) in enumerate(page_units, 1):
        ordered = (
            members
            if kind == "prose"
            else sorted(members, key=lambda item: (item.bbox[1], item.bbox[0]))
        )
        annotations = _label_annotations(kind, ordered)
        annotations.extend(_marker_annotations(ordered))
        paragraphs = []
        for block in ordered:
            if normalized := normalize_reading_text(block.text):
                paragraphs.append(normalized)
        text = "\n\n".join(paragraphs)
        if not text:
            continue
        units.append(
            {
                "id": (
                    f"{document_id}:pdf-page:{page_number:06d}:"
                    f"region:{region_index:03d}"
                ),
                "document_id": document_id,
                "source_id": source_id,
                "title": title,
                "contents": text,
                "content_kind": kind,
                "annotations": annotations,
                "quality_flags": _quality_flags(text, kind),
                "locator": locator,
            }
        )
    return units, False, removed_margin_blocks


def _extract_pdf(
    source: SourceFile,
    digest: str | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    digest = digest or sha256_file(source.path)
    document_id = _document_id(source, digest)
    try:
        document = pymupdf.open(source.path)
    except Exception as exc:
        raise ExtractionError(f"Cannot open PDF source: {source.path}") from exc

    try:
        if document.needs_pass:
            raise ExtractionError(
                f"Password-protected PDF is unsupported: {source.path}"
            )
        metadata = document.metadata or {}
        pages: list[tuple[list[_TextBlock], float, float]] = [
            (_page_blocks(page), float(page.rect.width), float(page.rect.height))
            for page in document
        ]
        automatic, provenance, warnings = _front_matter_identity(
            pages,
            metadata,
            source.path.stem,
            declared_language=_pdf_declared_language(document),
        )
        record = _base_metadata(
            source=source,
            document_id=document_id,
            digest=digest,
            automatic=automatic,
            provenance=provenance,
            warnings=warnings,
        )
        repeated_margins = _repeated_margin_signatures(pages)
        units: list[dict[str, Any]] = []
        empty_pages = 0
        removed_margin_blocks = 0
        for page_index, page in enumerate(document):
            page_number = page_index + 1
            blocks, page_width, page_height = pages[page_index]
            page_units, empty, removed = _pdf_page_units(
                page,
                page_number=page_number,
                blocks=blocks,
                page_width=page_width,
                page_height=page_height,
                repeated_margins=repeated_margins,
                document_id=document_id,
                source_id=source.source_id,
                title=str(record["title"]),
            )
            units.extend(page_units)
            empty_pages += int(empty)
            removed_margin_blocks += removed

        record.update(
            {
                "physical_pages": document.page_count,
                "extracted_units": len(units),
                "empty_units": empty_pages,
                "removed_repeated_margin_blocks": removed_margin_blocks,
            }
        )
    finally:
        document.close()

    if not units:
        raise ExtractionError(
            f"PDF produced no text; OCR may be required: {source.path}"
        )
    return record, units


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
    """Read a conservative title/byline fallback from the first visible sections."""

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
    """Return the visible text of the first few spine documents."""

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


def _extract_epub(
    source: SourceFile,
    digest: str | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    digest = digest or sha256_file(source.path)
    record, spine_items = prepare_epub_extraction(source, digest)
    units: list[dict[str, Any]] = []
    empty_sections = 0
    for spine_index in range(spine_items):
        batch, empty = extract_epub_spine_item(source, record, spine_index)
        units.extend(batch)
        empty_sections += int(empty)
    record["extracted_units"] = len(units)
    record["empty_units"] = empty_sections
    if not units:
        raise ExtractionError(f"EPUB produced no readable sections: {source.path}")
    return record, units


def pdf_page_count(source: SourceFile) -> int:
    """Return the physical page count after validating a PDF for extraction."""

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


def scan_pdf_pages(
    source: SourceFile,
    start_index: int,
    page_count: int,
) -> list[dict[str, Any]]:
    """Capture a consecutive page batch using one PDF document handle."""

    if start_index < 0:
        raise ValueError("start_index must be non-negative")
    if page_count <= 0:
        raise ValueError("page_count must be positive")

    try:
        document = pymupdf.open(source.path)
    except Exception as exc:
        raise ExtractionError(f"Cannot open PDF source: {source.path}") from exc
    try:
        if document.needs_pass:
            raise ExtractionError(
                f"Password-protected PDF is unsupported: {source.path}"
            )
        end_index = min(start_index + page_count, document.page_count)
        scans: list[dict[str, Any]] = []
        for page_index in range(start_index, end_index):
            page = document.load_page(page_index)
            blocks = _page_blocks(page)
            scans.append(
                {
                    "page_index": page_index,
                    "page_width": float(page.rect.width),
                    "page_height": float(page.rect.height),
                    "blocks": [
                        {
                            "number": block.number,
                            "bbox": list(block.bbox),
                            "lines": list(block.lines),
                            "text": block.text,
                            "font_size": block.font_size,
                        }
                        for block in blocks
                    ],
                }
            )
        return scans
    finally:
        document.close()


def _blocks_from_scan(scan: dict[str, Any]) -> list[_TextBlock]:
    return [
        _TextBlock(
            number=int(item["number"]),
            bbox=tuple(float(value) for value in item["bbox"]),
            lines=tuple(str(value) for value in item["lines"]),
            text=str(item["text"]),
            font_size=float(item["font_size"]),
        )
        for item in scan["blocks"]
    ]


def prepare_scanned_pdf(
    source: SourceFile,
    digest: str,
    page_scans: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[str]]:
    """Resolve PDF metadata and repeated margins from completed page scans."""

    try:
        document = pymupdf.open(source.path)
    except Exception as exc:
        raise ExtractionError(f"Cannot open PDF source: {source.path}") from exc
    try:
        if document.needs_pass:
            raise ExtractionError(
                f"Password-protected PDF is unsupported: {source.path}"
            )
        pages = [
            (
                _blocks_from_scan(scan),
                float(scan["page_width"]),
                float(scan["page_height"]),
            )
            for scan in page_scans
        ]
        automatic, provenance, warnings = _front_matter_identity(
            pages,
            document.metadata or {},
            source.path.stem,
            declared_language=_pdf_declared_language(document),
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
                "physical_pages": document.page_count,
                "extracted_units": 0,
                "empty_units": 0,
                "removed_repeated_margin_blocks": 0,
            }
        )
        return record, sorted(_repeated_margin_signatures(pages))
    finally:
        document.close()


def extract_scanned_pdf_pages(
    source: SourceFile,
    document_record: dict[str, Any],
    page_scans: list[dict[str, Any]],
    repeated_margins: list[str],
) -> list[tuple[int, list[dict[str, Any]], bool, int]]:
    """Extract a scanned page batch using one PDF document handle."""

    if not page_scans:
        raise ValueError("page_scans must not be empty")
    try:
        document = pymupdf.open(source.path)
    except Exception as exc:
        raise ExtractionError(f"Cannot open PDF source: {source.path}") from exc
    try:
        if document.needs_pass:
            raise ExtractionError(
                f"Password-protected PDF is unsupported: {source.path}"
            )
        repeated_margin_set = set(repeated_margins)
        batches: list[tuple[int, list[dict[str, Any]], bool, int]] = []
        for scan in page_scans:
            page_index = int(scan["page_index"])
            page = document.load_page(page_index)
            units, empty, removed = _pdf_page_units(
                page,
                page_number=page_index + 1,
                blocks=_blocks_from_scan(scan),
                page_width=float(scan["page_width"]),
                page_height=float(scan["page_height"]),
                repeated_margins=repeated_margin_set,
                document_id=str(document_record["document_id"]),
                source_id=str(document_record["source_id"]),
                title=str(document_record["title"]),
            )
            batches.append((page_index, units, empty, removed))
        return batches
    finally:
        document.close()


def prepare_epub_extraction(
    source: SourceFile,
    digest: str,
) -> tuple[dict[str, Any], int]:
    """Resolve EPUB metadata and return its deterministic spine work count."""

    try:
        book = epub.read_epub(str(source.path), options={"ignore_ncx": True})
    except Exception as exc:
        raise ExtractionError(f"Cannot open EPUB source: {source.path}") from exc
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


def _epub_fragment(tag: Tag) -> tuple[str, str] | None:
    """Return an existing XHTML fragment and its source attribute verbatim."""

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
    """Return a deterministic, human-inspectable path within one XHTML item."""

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


def _epub_semantic_blocks(
    soup: BeautifulSoup,
) -> list[tuple[Tag, tuple[str, str] | None]]:
    """Return outermost semantic blocks and a nearest standalone anchor."""

    tags = list(soup.find_all(True))
    positions = {id(tag): index for index, tag in enumerate(tags)}
    blocks = [
        tag
        for tag in tags
        if str(tag.name).casefold() in _EPUB_SEMANTIC_ELEMENTS
        and not any(
            isinstance(parent, Tag)
            and str(parent.name).casefold() in _EPUB_SEMANTIC_ELEMENTS
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
    """Split visible block text at nested XHTML anchors without marker leakage."""

    segments: list[tuple[str, tuple[str, str, str] | None]] = []
    parts: list[str] = []
    active_anchor = initial_anchor

    def flush() -> None:
        text = normalize_reading_text("".join(parts))
        parts.clear()
        if text:
            segments.append((text, active_anchor))

    for node in element.descendants:
        if isinstance(node, Comment):
            continue
        if isinstance(node, Tag):
            anchor = _epub_fragment(node)
            if anchor is not None:
                flush()
                active_anchor = (anchor[0], "nested", anchor[1])
            name = str(node.name).casefold()
            if name == "br":
                parts.append("\n")
            elif name in _EPUB_CONTENT_ELEMENTS:
                parts.append("\n\n")
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
) -> tuple[list[dict[str, Any]], bool]:
    """Extract one EPUB spine entry while preserving its original locator."""

    try:
        book = epub.read_epub(str(source.path), options={"ignore_ncx": True})
    except Exception as exc:
        raise ExtractionError(f"Cannot open EPUB source: {source.path}") from exc
    spine_entry = book.spine[spine_index]
    item_id = spine_entry[0] if isinstance(spine_entry, tuple) else spine_entry
    item = book.get_item_with_id(item_id)
    if item is None or item.get_type() != ebooklib.ITEM_DOCUMENT:
        return [], False
    soup = BeautifulSoup(item.get_content(), "html.parser")
    for unwanted in soup(["script", "style", "nav"]):
        unwanted.decompose()
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
        block_index += 1
        locator = _epub_locator(
            section_index=spine_index + 1,
            section_title=section_title,
            href=href,
            element=element,
            block_index=block_index,
            anchor=anchor,
        )
        units.append(
            {
                "id": (
                    f"{document_id}:epub-section:{spine_index + 1:06d}:"
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
        element_name = str(element.name).casefold()
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
    return units, not units


def extract_sources(
    sources: tuple[SourceFile, ...],
    source_digests: dict[str, str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return indexed documents and cleaned semantic units."""

    documents: list[dict[str, Any]] = []
    units: list[dict[str, Any]] = []
    for source in sources:
        digest = (source_digests or {}).get(source.source_relative_path)
        if source.extension == ".pdf":
            document, extracted = _extract_pdf(source, digest)
        elif source.extension == ".epub":
            document, extracted = _extract_epub(source, digest)
        else:  # pragma: no cover
            raise ExtractionError(f"Unsupported source format: {source.path}")
        retained: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        for unit in extracted:
            reasons = text_health_reasons(str(unit.get("contents") or ""))
            if reasons:
                rejected.append(
                    {
                        "unit_id": str(unit.get("id") or ""),
                        "locator": dict(unit.get("locator") or {}),
                        "reasons": reasons,
                    }
                )
            else:
                retained.append(unit)
        if rejected:
            document["excluded_corrupt_unit_count"] = len(rejected)
            document["excluded_corrupt_units"] = rejected
            document["metadata_warnings"] = list(
                dict.fromkeys(
                    [
                        *document.get("metadata_warnings", []),
                        "corrupt_extraction_units_excluded",
                    ]
                )
            )
        else:
            document["excluded_corrupt_unit_count"] = 0
            document["excluded_corrupt_units"] = []
        document["extracted_units"] = len(retained)
        if not retained:
            raise ExtractionError(
                "Source produced no readable English-oriented text after unhealthy "
                f"extraction units were excluded: {source.path}"
            )
        documents.append(document)
        units.extend(retained)
    return documents, units
