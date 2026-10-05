"""Forensics on one string of extracted text, and the verdicts built from them.

A reason is evidence about the text itself, never about the source it came from.
This module reads no file, opens no document, reaches no network, and imports
nothing from this package.

`text_normalization` owns the text every verdict is computed over. `extraction`
removes what this module is confident is not semantic text, withholds a unit
whose text carries a reason, and refuses a source whose own units agree that
there is nothing to retrieve from it.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Sequence

from .text_normalization import normalize_inline_text

_MOJIBAKE_MARKERS = ("â€", "ï¿½", "ðŸ")
_MOJIBAKE_LATIN1_PAIR = re.compile(r"(?:Ã|Â)[\u0080-\u00bf]")


def _script_family(character: str) -> str | None:
    """Script family for alphabetic text, resolved without a dependency."""

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
    """The corruption evidence that withholds extraction text.

    Only incoherent output is withheld: replacement characters, private-use or
    unassigned code points, and known damaged encoding sequences. Script mixing and
    non-Latin dominance are notes, so quotations stay retrievable.
    """

    return _text_signals(value)[0]


def text_script_notes(value: str) -> list[str]:
    """Advisory non-Latin or mixed-script notes that never withhold text."""

    return _text_signals(value)[1]


def has_searchable_alphanumeric_content(value: str) -> bool:
    return any(character.isalnum() for character in normalize_inline_text(value))


def text_health_reasons(value: str) -> list[str]:
    reasons = text_corruption_reasons(value)
    normalized = normalize_inline_text(value)
    if normalized and not has_searchable_alphanumeric_content(normalized):
        reasons.append("symbol_only")
    return reasons


# Retrieval rejects a candidate for exactly two reasons, and both are properties
# of the chunk text plus its stored quality flags rather than of the query. They
# are computed once when the artifact lookup is built and stored as a bitmask,
# which removes the per-query text scans from the candidate gate.
CHUNK_FLAG_CORRUPT_TEXT = 1
CHUNK_FLAG_EXTRACTION_ARTIFACT = 2

# The token `extraction` stores against a chunk it could not read as prose. It is
# written by one module and read by this one, so it is named here and nowhere else.
EXTRACTION_ARTIFACT_TOKEN = "extraction_artifact"


# ---------------------------------------------------------------------------
# Removing text that is not the document's argument
# ---------------------------------------------------------------------------
#
# A unit's text reaches a chunker, and a chunk is what a reader is shown. Running
# furniture, a reference list, and a note apparatus are all prose-shaped, so no
# corruption signal can tell them from evidence, and they must be recognised by
# their own shape instead. Each rule below is deliberately narrow: it fires only
# where the evidence is structural rather than topical, so a passage that argues
# about references, quotes a bibliography, or prints a numbered formula keeps its
# text. Every rule names a closed set of headings rather than a topic word, which
# is what keeps a foreign-language source out of the rule.

_REFERENCE_HEADINGS = frozenset(
    {
        "references",
        "reference",
        "bibliography",
        "works cited",
        "literature cited",
        "references and notes",
        "notes and references",
        "citations",
        "reference list",
    }
)
_NOTE_HEADINGS = frozenset(
    {
        "notes",
        "footnotes",
        "endnotes",
    }
)
# A note is printed with its marker at the start of its own line: a star, a dagger,
# a pilcrow, or its own number. The number may carry a period or a parenthesis
# after it or neither, because a publisher sets it either way; it may not be a year,
# so a paragraph that opens with one of those is not a note marker.
_NOTE_MARKER = re.compile(
    r"^\s*(?:\*|∗|†|‡|§|¶|\d{1,3}[.)]?)(?:\s|$)",
    re.UNICODE,
)
# A bibliographic marker is a year, a DOI, or a URL the publisher printed, or the
# page range a list gives. A year in brackets is only one of these when the
# brackets hold the year and nothing else and the element ends there: a citation
# in running prose puts the author's name inside the brackets, as in
# (Smith 2019, 14), and a sentence that opens by citing reads "Jones, A. (2020)
# argues that ...", where the bracket does not end the element.
_PARENTHETICAL_YEAR = re.compile(
    r"[([]\s*(?:18|19|20|21)\d{2}[a-z]?"
    r"(?:\s*[,;:]\s*(?:18|19|20|21)\d{2}[a-z]?)*\s*[)\]]\s*[.,;]"
)
_DOI_OR_URL = re.compile(
    r"(?:\b10\.\d{4,9}/\S|\bdoi\s*[:.]|\bhttps?://|\bwww\.)",
    re.IGNORECASE,
)
_PAGE_RANGE = re.compile(r"\bpp?\.\s*\d+(?:\s*[-–—]\s*\d+)?", re.IGNORECASE)
# An entry opens with an author, and an author is a surname with the author's own
# initials after it: `Smith, J. A.` or `J. A. Smith`. That is what separates an
# entry from an argument paragraph, because an argument opens with a word and a
# comma just as often — "However, the committee rejected the plan", "Later, the
# council asked" — and a surname followed by any word at all would read both as
# entries and take a chapter's prose with them. A bibliography that spells given
# names out is kept rather than removed, which is the safe direction. A line the
# publisher printed as a bare locator is an entry whatever it follows, because no
# sentence begins with one.
_AUTHOR_INITIALS = r"(?:[^\W\d_]\.[\s]?){1,4}"
_BARE_LOCATOR = r"(?:https?://|www\.|10\.\d{4,9}/)\S+"
_AUTHOR_LEAD = re.compile(
    rf"^\s*(?:[^\W\d_][\w'’\-]*[,\s]+{_AUTHOR_INITIALS}"
    rf"|{_AUTHOR_INITIALS}[^\W\d_][\w'’\-]*"
    rf"|{_BARE_LOCATOR})",
    re.UNICODE,
)
# An entry is a line of a list, so it is short. The cap is what keeps a paragraph
# of argument that happens to cite (Smith 2019) from being read as an entry and
# taking a chapter's argument with it; a real entry that runs past the cap is kept
# rather than removed, which is the safe direction.
_MAXIMUM_ENTRY_CHARACTERS = 400
# A heading is a short line that does not end as a sentence does. A paragraph
# opening "References to the earlier work show ..." is body text that happens to
# open with a word, and matching the word alone would delete the argument that
# follows it.
_HEADING_MAXIMUM_CHARACTERS = 80
_SENTENCE_END = ".,;!?"


def _heading_key(line: str) -> str:
    """The comparable form of a line that may be a section heading."""

    normalized = normalize_inline_text(line)
    return normalized.rstrip(":.").strip().casefold()


def _is_heading_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped or len(stripped) > _HEADING_MAXIMUM_CHARACTERS:
        return False
    return stripped[-1] not in _SENTENCE_END


def opens_with_note_marker(line: str) -> bool:
    """Whether a line opens with the marker its number or note is printed with.

    A note apparatus is printed with a marker on every one of its paragraphs, and
    an argument paragraph is not. The test is on the line rather than the
    paragraph because a page sets one note per line.
    """

    return bool(_NOTE_MARKER.match(line))


def is_reference_entry(value: str) -> bool:
    """Whether a paragraph reads as an entry of a reference list.

    Two things are required and neither is a topic word: the paragraph opens with
    an author or a bare locator, and it carries a bibliographic marker the
    publisher printed — a year in brackets that ends its element, a DOI, a URL, or
    a page range. A reference list is the one place a document reliably prints such
    things, and requiring both is what keeps an ordinary paragraph out of the
    answer: an argument cites, so it carries a year and a name too, but it opens
    with a word rather than with an author's initials, and it puts the name inside
    the brackets.
    """

    text = value.strip()
    if not text or len(text) > _MAXIMUM_ENTRY_CHARACTERS:
        return False
    if not _AUTHOR_LEAD.match(text):
        return False
    return bool(
        _PARENTHETICAL_YEAR.search(text)
        or _DOI_OR_URL.search(text)
        or _PAGE_RANGE.search(text)
    )


def non_argument_section_heading(value: str) -> str | None:
    """The kind of non-argument section a paragraph opens, if any.

    Returns `"bibliography"`, `"footnotes"`, or `None`. The heading has to be the
    whole line, and the line has to be short and free of sentence punctuation, so
    a paragraph that opens "References to the earlier work show ..." is body text
    and opens nothing.
    """

    stripped = value.strip()
    if not _is_heading_line(stripped):
        return None
    key = _heading_key(stripped)
    if key in _REFERENCE_HEADINGS:
        return "bibliography"
    if key in _NOTE_HEADINGS:
        return "footnotes"
    return None


def _continuation_entry(paragraph: str, kind: str) -> bool:
    """Whether a paragraph continues a non-argument section opened earlier.

    The test is the same one the heading's own section is judged by, so a section
    is only ever removed as far as it keeps looking like itself. The first
    paragraph that does not is where the section ends.
    """

    return (
        is_reference_entry(paragraph)
        if kind == "bibliography"
        else opens_with_note_marker(paragraph)
    )


def non_argument_removal_flags(paragraphs: Sequence[str]) -> list[str | None]:
    """Mark which paragraphs of a sequence belong to a confirmed non-argument section.

    Returns one entry per paragraph: the kind removed (`"bibliography"` or
    `"footnotes"`), or `None` for a paragraph that stays.

    The sequence is the whole source in reading order, not one unit, because a
    section does not fit in a unit: a book's reference list may open with a
    heading on its own page and run over the next three, and an EPUB emits its
    heading as one element and each entry as another, so no single unit ever holds
    a heading and its entries together.

    A section opens at a heading in the closed set and is marked only while the
    paragraphs after it keep its own shape. The first paragraph that does not ends
    it, which is also where the next argument heading or a new chapter ends it, and
    the sequence's end ends it. A heading is marked only once a paragraph of the
    section's own shape has confirmed it, so a "References" heading above ordinary
    prose marks nothing at all, and a citation inside an argument is never inside a
    section to begin with.
    """

    flags: list[str | None] = [None] * len(paragraphs)
    heading_index: int | None = None
    kind: str | None = None
    confirmed = False

    def close() -> None:
        nonlocal heading_index, kind, confirmed
        if heading_index is not None and confirmed:
            flags[heading_index] = kind
        heading_index = None
        kind = None
        confirmed = False

    for index, paragraph in enumerate(paragraphs):
        if not paragraph.strip():
            continue
        opened = non_argument_section_heading(paragraph)
        if opened is not None:
            close()
            heading_index, kind = index, opened
            continue
        if kind is not None and _continuation_entry(paragraph, kind):
            flags[index] = kind
            confirmed = True
            continue
        close()
    close()
    return flags


def strip_non_argument_sections(
    paragraphs: Sequence[str],
) -> tuple[list[str], tuple[str, ...]]:
    """Remove confirmed reference and note sections from a sequence of paragraphs.

    Returns the retained paragraphs and the kinds removed, in the order they first
    fired. `extraction.strip_non_argument_units` is the same decision over a whole
    source's units and is what both build paths call; this is the paragraph-level
    answer to the same question.
    """

    flags = non_argument_removal_flags(paragraphs)
    kept = [text for text, removed in zip(paragraphs, flags) if removed is None]
    removed_kinds = [kind for kind in flags if kind is not None]
    return kept, tuple(dict.fromkeys(removed_kinds))


# ---------------------------------------------------------------------------
# Refusing a source whose own text says it is unreadable
# ---------------------------------------------------------------------------
#
# A unit is withheld on its own evidence and the source survives without it: a
# corrupt page in an otherwise readable paper is a hole, not a rejection. A
# source is a different question, because the evidence is now about the whole
# file rather than one page, and the remedy is a decision the reader makes about
# the file. Every reason below is one a reader can check by opening the source.

# No unit carried searchable text, so the file holds images or drawings only.
SOURCE_REASON_NO_TEXT = "no_readable_text"
# Every page was a scan, so the file has no text layer anywhere.
SOURCE_REASON_NO_TEXT_LAYER = "no_text_layer"
# Almost every unit was withheld, so what survived is not the document.
SOURCE_REASON_ALMOST_ALL_WITHHELD = "almost_all_units_withheld"
# The text is dominated by symbols, digits, and layout marks rather than words.
SOURCE_REASON_ARTIFACT_LADEN = "artifact_laden_text"
# Cleaning removed more of the file than it kept, so its own text cannot be
# trusted to say what the document argued.
SOURCE_REASON_UNSAFE_TO_CLEAN = "unsafe_to_clean"
# No retained unit held a letter at all, so what the file gave is digits, table
# rules, and layout marks. Digits are searchable text and a numeric table is a
# document, so this fires only where the whole file is numbers: a page of
# numbers beside a labelled paragraph has letters somewhere in the source, and
# that source is kept.
SOURCE_REASON_NO_LETTER_TEXT = "no_letter_text"

# Below this share of retained units the source is a document with damaged
# pages rather than a damaged document. The existing policy withholds a corrupt
# unit and keeps the source; refusing the file instead would make one bad scan
# page cost a reader the whole paper, so the bar sits where nearly everything
# has to be unreadable before the file is refused.
_MINIMUM_RETAINED_PERCENT = 10
# Above this share of a file's characters the cleanup removed — running furniture,
# a sidebar, a note block, a confirmed section — the heuristics that removed them
# cannot be trusted to have left the argument. The threshold sits above half
# because a reference list or a note apparatus can legitimately be most of a short
# work — a bibliography is the whole of a reading list — and this must refuse only a
# file whose own text no longer reads as a document.
_MAXIMUM_CLEANED_PERCENT = 50
# Below this many removed characters the share is not evidence about anything.
_MINIMUM_CLEANED_CHARACTERS = 500


def source_health_reasons(
    *,
    unit_count: int,
    retained_count: int,
    withheld_reasons: Counter[str],
    removed_characters: int = 0,
    kept_characters: int = 0,
    image_only_pages: int = 0,
    physical_pages: int = 0,
    letter_characters: int | None = None,
) -> list[str]:
    """The reasons a whole source is refused, from evidence about the whole file.

    A source with at least one readable unit is never refused for what it lost:
    the unit rule already withholds what it cannot use, and a reader who wants the
    file indexed anyway can exclude the parts they do not want instead. The three
    exceptions are a file whose text layer is absent, which no amount of
    excluding makes readable, a file whose cleaning removed more than it kept,
    where what remains cannot be said to be the document's argument, and a file
    whose every retained unit is digits and layout marks, which is a table with
    nothing around it to say what the numbers are.

    `letter_characters` is the count of letters in the retained units' text, and
    it is `None` when the caller has not measured it. Only a measured zero is
    evidence, so an unmeasured call makes no claim about letters.

    A page is judged a scan by the extractor, which can see the page's images; a
    share of digits is not a scan. A file with only some scanned pages is not
    refused for them — the readable pages are the document, and the scans are
    recorded on it.
    """

    if unit_count <= 0 or retained_count <= 0:
        if physical_pages > 0 and image_only_pages >= physical_pages:
            return [SOURCE_REASON_NO_TEXT_LAYER]
        return [SOURCE_REASON_NO_TEXT]

    reasons: list[str] = []
    if retained_count * 100 < unit_count * _MINIMUM_RETAINED_PERCENT:
        reasons.append(SOURCE_REASON_ALMOST_ALL_WITHHELD)
        if withheld_reasons.get("symbol_only", 0) * 100 >= unit_count * 50:
            reasons.append(SOURCE_REASON_ARTIFACT_LADEN)
    if letter_characters == 0:
        reasons.append(SOURCE_REASON_NO_LETTER_TEXT)
    if physical_pages > 0 and image_only_pages >= physical_pages:
        reasons.append(SOURCE_REASON_NO_TEXT_LAYER)

    total = kept_characters + removed_characters
    if (
        removed_characters >= _MINIMUM_CLEANED_CHARACTERS
        and removed_characters * 100 > total * _MAXIMUM_CLEANED_PERCENT
    ):
        reasons.append(SOURCE_REASON_UNSAFE_TO_CLEAN)
    return reasons


def chunk_health_flags(text: str, *, quality_flags: object = None) -> int:
    """A precomputed verdict: an extraction artifact, or has no searchable
    alphanumeric content. Mirrors the query-time check, so counters cannot change.
    """
    flags = 0
    if text_corruption_reasons(text):
        flags |= CHUNK_FLAG_CORRUPT_TEXT
    stored = {str(item) for item in quality_flags or ()}
    if EXTRACTION_ARTIFACT_TOKEN in stored or not has_searchable_alphanumeric_content(
        text
    ):
        flags |= CHUNK_FLAG_EXTRACTION_ARTIFACT
    return flags
