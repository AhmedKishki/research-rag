"""Forensics on one string of extracted text, and the verdicts built from them.

A reason describes text or its declared structure, never its topical relevance.
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
from collections.abc import Iterator, Mapping, Sequence

from .text_normalization import (
    is_control_character,
    is_known_formatting_glyph,
    normalize_inline_text,
)

_MOJIBAKE_MARKERS = ("â€", "ï¿½", "ðŸ")
_MOJIBAKE_LATIN1_PAIR = re.compile(r"(?:Ã|Â)[\u0080-\u00bf]")

# The code-point categories no character map is supposed to produce in prose: a
# private-use codepoint, an unassigned codepoint, and a lone surrogate. Together
# with a replacement character and a stray C0 control they are the characters a
# passage's unreadable share is counted over. A private-use codepoint that is a
# documented formatting glyph (a symbol font's bullet, a tall delimiter piece) is
# not counted: it is what the source printed, not corruption.
_CORRUPT_CHARACTER_CATEGORIES = frozenset({"Co", "Cn", "Cs"})


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


def _counts_as_unreadable(character: str) -> bool:
    """Whether one character is evidence a passage is not readable prose.

    The categories no character map produces in prose count, except a private-use
    code point that is a documented formatting glyph: a symbol font's bullet and
    a tall delimiter piece are what the source drew, so they never withhold text.
    A replacement character and a stray control character count as well.
    """

    return (
        character == "\ufffd"
        or (
            unicodedata.category(character) in _CORRUPT_CHARACTER_CATEGORIES
            and not is_known_formatting_glyph(character)
        )
        or is_control_character(character)
    )


def _text_signals(value: str) -> tuple[list[str], list[str]]:
    raw = unicodedata.normalize("NFC", value)
    normalized = normalize_inline_text(raw)
    if not normalized:
        return [], []
    replacement_count = normalized.count("\ufffd")
    private_or_unassigned = sum(
        unicodedata.category(character) in {"Co", "Cn", "Cs"}
        and not is_known_formatting_glyph(character)
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

    Only incoherent output is withheld: replacement characters, unassigned or
    surrogate code points, an unrecognised private-use glyph, and known damaged
    encoding sequences. A documented formatting glyph from a symbol face and
    script mixing are not corruption, so a quotation and a bulleted passage stay
    retrievable.
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


def _pdf_lexical_spans(value: str) -> Iterator[tuple[int, int]]:
    """Keep whitespace-free scripts local without splitting Latin words."""

    unspaced = {"cjk", "japanese", "thai", "lao", "khmer", "myanmar"}
    for match in re.finditer(r"\S+", value):
        position = match.start()
        while position < match.end():
            end = position + 1
            if _script_family(value[position]) in unspaced:
                while end < match.end() and unicodedata.category(value[end]).startswith(
                    "M"
                ):
                    end += 1
            else:
                while end < match.end() and _script_family(value[end]) not in unspaced:
                    end += 1
            yield position, end
            position = end


def clean_damaged_pdf_passage(value: str) -> tuple[str, int, list[dict[str, object]]]:
    """Remove damaged lexical tokens, without inventing or completing a word.

    The caller first attempts reversible encoding repair and native extraction.
    Unknown glyphs inside a word invalidate that entire token, not just the
    glyph. An explicit gap prevents the remaining words from looking contiguous.
    Whitespace-free scripts are divided at script and combining-cluster boundaries,
    so one bad glyph cannot invalidate an otherwise readable CJK paragraph.
    Offsets refer to the supplied, post-repair paragraph. Lost characters exclude
    whitespace and include all readable characters removed with the bad token.
    """

    spans: list[dict[str, object]] = []
    pieces: list[str] = []
    position = 0
    lost = 0
    for start, end in _pdf_lexical_spans(value):
        text = value[start:end]
        reasons = text_corruption_reasons(text)
        if any(_counts_as_unreadable(character) for character in text):
            reasons = list(dict.fromkeys([*reasons, "unreadable_characters"]))
        if not reasons:
            continue
        pieces.append(value[position:start])
        pieces.append(" [...] ")
        size = sum(not character.isspace() for character in text)
        lost += size
        spans.append(
            {
                "start": start,
                "end": end,
                "lost_characters": size,
                "reasons": reasons,
            }
        )
        position = end
    if not spans:
        return value, 0, []
    pieces.append(value[position:])
    cleaned = normalize_inline_text("".join(pieces))
    cleaned = re.sub(r"(?:\[\.\.\.\]\s*){2,}", "[...] ", cleaned).strip()
    return cleaned, lost, spans


def passage_health_reasons(
    value: str, *, maximum_unclean_percent: float | None = None
) -> list[str]:
    """One passage's verdict: its mechanical reasons, and its unreadable share.

    The mechanical reasons are `text_health_reasons`' own: replacement
    characters, private, unassigned, or surrogate code points, a known damaged
    encoding sequence, and text with no alphanumeric content. They are always
    returned, so a share of unreadable text never waives them, not even at 100.

    `maximum_unclean_percent` adds `unclean_text` when the share of characters
    that are a replacement, a private, unassigned, or surrogate code point, or a
    stray control exceeds it. `None` makes no such claim and returns the
    mechanical reasons alone. Quality is judged here, per passage: a readable
    unit survives whatever share the whole source lost, because the unit rule
    already withholds what it cannot use.
    """

    reasons = text_health_reasons(value)
    if maximum_unclean_percent is not None:
        total = len(value)
        if total > 0:
            unreadable = sum(_counts_as_unreadable(character) for character in value)
            if unreadable * 100 > total * maximum_unclean_percent:
                reasons.append(SOURCE_REASON_UNCLEAN)
    return list(dict.fromkeys(reasons))


# Retrieval always rejects a candidate for two reasons, and rejects an apparatus
# passage when a project asks it to. All three describe chunk text, stored quality
# flags and declared structural context rather than the query. They are computed once
# when the artifact lookup is built and stored as a bitmask, which removes the
# per-query text scans from the candidate gate.
CHUNK_FLAG_CORRUPT_TEXT = 1
CHUNK_FLAG_EXTRACTION_ARTIFACT = 2
CHUNK_FLAG_APPARATUS = 4
# The flags no reader is ever shown, whatever a project's settings say.
CHUNK_FLAGS_UNREADABLE = CHUNK_FLAG_CORRUPT_TEXT | CHUNK_FLAG_EXTRACTION_ARTIFACT

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
        else bool(_NOTE_MARKER.match(paragraph))
    )


def non_argument_removal_flags(
    paragraphs: Sequence[str], *, labels: Sequence[str] | None = None
) -> list[str | None]:
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

    Docling callers supply one structural label per paragraph. Only declared
    headings open sections, and captions, formulas, tables, code, footnotes, and
    unknown item kinds end them without losing their content. A declared reference
    still needs an entry's shape. Marked notes need bibliographic content as well,
    so explanatory notes are not discarded merely because they carry a number.
    Unlabelled EPUB and MOBI callers retain their existing section rules.
    """

    if labels is not None and len(labels) != len(paragraphs):
        raise ValueError("Structural labels must match the paragraph sequence")
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
        label = labels[index].casefold() if labels is not None else None
        heading = label in {"section_header", "title"}
        if (
            label is not None
            and not heading
            and label
            not in {
                "text",
                "reference",
                "list_item",
                "list",
            }
        ):
            close()
            continue
        opened = (
            non_argument_section_heading(paragraph)
            if label is None or heading
            else None
        )
        if opened is not None:
            close()
            heading_index, kind = index, opened
            continue
        if heading:
            close()
            continue
        continuation = kind is not None and _continuation_entry(paragraph, kind)
        if continuation and label is not None and kind == "footnotes":
            continuation = is_reference_entry(_NOTE_MARKER.sub("", paragraph, count=1))
        if continuation:
            flags[index] = kind
            confirmed = True
            continue
        close()
        if label == "reference" and is_reference_entry(paragraph):
            flags[index] = "bibliography"
    close()
    return flags


# ---------------------------------------------------------------------------
# Refusing a source whose own text says it is unreadable
# ---------------------------------------------------------------------------
#
# PDF loss is measured over substantive document text after local cleaning.
# EPUB keeps its existing aggregate policy.

# No unit carried searchable text, so the file holds images or drawings only.
SOURCE_REASON_NO_TEXT = "no_readable_text"
# Every page was a scan, so the file has no text layer anywhere.
SOURCE_REASON_NO_TEXT_LAYER = "no_text_layer"
# More of a passage's text was unreadable than a caller accepts. It is a
# PDF passage-level reason; EPUB retains its source-level use.
SOURCE_REASON_UNCLEAN = "unclean_text"
# Aggregate reasons apply only to the legacy EPUB policy.
SOURCE_REASON_ALMOST_ALL_WITHHELD = "almost_all_units_withheld"
SOURCE_REASON_ARTIFACT_LADEN = "artifact_laden_text"
SOURCE_REASON_UNSAFE_TO_CLEAN = "unsafe_to_clean"
SOURCE_REASON_NO_LETTER_TEXT = "no_letter_text"


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
    withheld_characters: int = 0,
    maximum_unclean_percent: float | None = None,
    passage_level: bool = False,
) -> list[str]:
    """Refuse an empty PDF or apply the unchanged EPUB source-level policy.

    With `passage_level=True`, PDF passages were cleaned individually, and only
    their cumulative substantive loss can refuse a remaining readable document.
    The argument name remains compatible with existing callers. The default
    preserves the aggregate checks used by EPUB ingestion.
    """

    if unit_count <= 0 or retained_count <= 0:
        if physical_pages > 0 and image_only_pages >= physical_pages:
            return [SOURCE_REASON_NO_TEXT_LAYER]
        return [SOURCE_REASON_NO_TEXT]
    if passage_level:
        if (
            maximum_unclean_percent is not None
            and withheld_characters > 0
            and withheld_characters * 100 > kept_characters * maximum_unclean_percent
        ):
            return [SOURCE_REASON_UNCLEAN]
        return []
    reasons: list[str] = []
    if retained_count * 100 < unit_count * 10:
        reasons.append(SOURCE_REASON_ALMOST_ALL_WITHHELD)
        if withheld_reasons.get("symbol_only", 0) * 100 >= unit_count * 50:
            reasons.append(SOURCE_REASON_ARTIFACT_LADEN)
    if letter_characters == 0:
        reasons.append(SOURCE_REASON_NO_LETTER_TEXT)
    if physical_pages > 0 and image_only_pages >= physical_pages:
        reasons.append(SOURCE_REASON_NO_TEXT_LAYER)
    if (
        maximum_unclean_percent is not None
        and withheld_characters > 0
        and withheld_characters * 100 > kept_characters * maximum_unclean_percent
    ):
        reasons.append(SOURCE_REASON_UNCLEAN)
    total = kept_characters + removed_characters
    if removed_characters >= 500 and removed_characters * 100 > total * 50:
        reasons.append(SOURCE_REASON_UNSAFE_TO_CLEAN)
    return reasons


# ---------------------------------------------------------------------------
# Recognising the scholarly apparatus a search may leave out
# ---------------------------------------------------------------------------
#
# Removal at ingestion stops where a section loses its shape, so a reference list
# that spells given names out, a run of citation-only notes, a back-of-book index
# and a copyright page reach the index whole. Each carries the names, titles and
# years a query asks about and no argument to cite, so it wins exactly the search
# the corpus cannot answer. `apparatus_kind` names one passage's kind from its own
# text and optional structural context; a project decides whether search leaves it out.
#
# Argument is told apart by grammar rather than topic. The pronouns, auxiliaries
# and conjunctions an argument cannot do without are about a sixth of its words
# and almost none of a reference list's, so only a passage nearly without them is
# judged, and then only by structure: citation markers among mostly capitalised
# words, page lists, or a copyright page's boilerplate. Two guards keep what is
# still evidence: twenty words of argument sentences anywhere in the passage, which
# is how a conclusion running into its references survives, and a table of
# measurements. A few function words of the languages a corpus quotes keep their
# prose out of the rule; a foreign reference list is still recognised.

APPARATUS_BIBLIOGRAPHY = "bibliography"
APPARATUS_INDEX = "index"
APPARATUS_FRONT_MATTER = "front_matter"
APPARATUS_HEADING = "heading"
_INDEX_HEADINGS = frozenset(
    {"index", "subject index", "name index", "index of subjects", "index of names"}
)


def apparatus_section_kind(value: str) -> str | None:
    """Recognise a declared section title for retrieval, never ingestion removal."""

    return APPARATUS_INDEX if _heading_key(value) in _INDEX_HEADINGS else None


_WORD = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)?", re.UNICODE)
_ARGUMENT_WORDS = frozenset(
    {
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "am",
        "that",
        "which",
        "who",
        "whom",
        "whose",
        "this",
        "these",
        "those",
        "it",
        "its",
        "we",
        "our",
        "us",
        "they",
        "them",
        "their",
        "he",
        "him",
        "his",
        "she",
        "her",
        "not",
        "no",
        "but",
        "however",
        "because",
        "since",
        "although",
        "though",
        "while",
        "whereas",
        "if",
        "then",
        "than",
        "also",
        "only",
        "even",
        "can",
        "could",
        "would",
        "should",
        "must",
        "may",
        "might",
        "will",
        "shall",
        "do",
        "does",
        "did",
        "has",
        "have",
        "had",
        "there",
        "here",
        "what",
        "when",
        "where",
        "why",
        "how",
        "such",
        "so",
        "thus",
        "therefore",
        "hence",
        "yet",
        "very",
        "more",
        "most",
        "less",
        "much",
        "many",
        "all",
        "any",
        "each",
        "every",
        "both",
        "either",
        "neither",
        "one",
        "i",
        "you",
        "your",
        "my",
        "me",
        "ist",
        "sind",
        "war",
        "wird",
        "werden",
        "wurde",
        "hat",
        "haben",
        "dass",
        "nicht",
        "auch",
        "wir",
        "sie",
        "es",
        "ich",
        "est",
        "sont",
        "était",
        "nous",
        "ils",
        "elle",
        "pas",
        "qui",
        "son",
        "fue",
        "era",
        "está",
    }
)
# A sentence of argument is also built from articles and prepositions, which a
# title shares, so they count only toward recognising a sentence.
_FUNCTION_WORDS = _ARGUMENT_WORDS | frozenset(
    {
        "the",
        "of",
        "and",
        "to",
        "in",
        "a",
        "an",
        "for",
        "on",
        "with",
        "by",
        "as",
        "or",
        "from",
        "at",
        "into",
        "between",
        "within",
        "through",
        "about",
        "over",
        "under",
        "der",
        "die",
        "das",
        "und",
        "mit",
        "von",
        "zu",
        "den",
        "dem",
        "le",
        "la",
        "les",
        "et",
        "dans",
        "pour",
        "que",
        "une",
        "des",
        "du",
        "el",
        "los",
        "las",
        "y",
        "del",
        "en",
        "por",
        "con",
        "para",
        "una",
    }
)
# Argument words a passage must stay at or under for its structure to be judged.
_APPARATUS_MAXIMUM_ARGUMENT_SHARE = 0.06
# A copyright page sets legal sentences, so it is judged with more of them.
_FRONT_MATTER_MAXIMUM_ARGUMENT_SHARE = 0.12

_CITATION_YEAR = re.compile(r"(?<![\d.])(?:1[5-9]|20)\d{2}[a-z]?(?!\d)")
_SURNAME_INITIALS = re.compile(r"\b[A-Z][a-zà-ÿ'’\-]+,\s(?:[A-Z]\.\s?){1,3}")
_INITIALS_SURNAME = re.compile(r"\b(?:[A-Z]\.\s?){1,3}\s?[A-Z][a-zà-ÿ'’\-]{2,}")
_SURNAME_GIVEN = re.compile(
    r"\b[A-Z][a-zà-ÿ'’\-]+,\s[A-Z][a-zà-ÿ'’\-]+(?:\s[A-Z]\.)?[.,]"
)
_CITATION_PAGES = re.compile(r"\bpp?\.\s?\d+", re.IGNORECASE)
_CITATION_LOCATOR = re.compile(
    r"(?:\b10\.\d{4,9}/|\bdoi\b|https?://|www\.|arxiv)", re.IGNORECASE
)
_CITATION_VENUE = re.compile(
    r"\b(?:Press|Journal|Proceedings|Review|Quarterly|Conference|Publishers?|"
    r"Verlag|Editions?|eds?\.|vol\.|no\.|Vol\.|No\.|Trans\.|trans\.)"
)
_CITATION_NOTE = re.compile(r"\b(?:Ibid|ibid|op\.\s?cit|Cf\.|cf\.|See also|see also)\b")
# An entry opens with a surname, a comma, and given names or initials, and reaches
# its year within the line.
_CITATION_ENTRY = re.compile(
    r"(?:^|(?<=[.\n”\"’)])\s*)"
    r"[A-Z][\w'’\-]+,\s(?:[A-Z][\w'’\-]*\.?\s?){1,4}"
    r"[^\n]{0,120}?[(\s](?:1[5-9]|20)\d{2}[a-z]?[).,:]"
)
_CITATION_MINIMUM_CAPITAL_SHARE = 0.25
_CITATION_MINIMUM_DENSITY = 0.06
_CITATION_MINIMUM_ENTRIES = 2

_INDEX_ENTRY = re.compile(
    r"[^\W\d_][^,\n]{0,40},\s\d{1,4}(?:[–-]\d{1,4})?(?:,\s\d{1,4}(?:[–-]\d{1,4})?)+"
)
# Unlike the text-only index heuristic, a tiny entry needs a declared Index
# section. Match the entire entry, never a numeric tail inside prose or a table.
_INDEX_PAGE = r"\d{1,4}(?:\s*[–—-]\s*\d{1,4})?"
_CONTEXTUAL_INDEX_ENTRY = re.compile(
    rf"[^\W\d_][^\d\n,;:.!?=]{{0,80}}(?:,\s*|\s+)"
    rf"{_INDEX_PAGE}(?:\s*[,;]\s*{_INDEX_PAGE})*\s*",
    re.UNICODE,
)
_SPLIT_CITATION_AUTHOR = re.compile(
    r"^[A-Z][\w'’\-]+,\s*(?:[A-Z][\w'’\-]*\.?\s*){1,4}$"
)
_SPLIT_CITATION_YEAR = re.compile(r"^\s*\(?(?:1[5-9]|20)\d{2}[a-z]?[).,:]\s*")
_CITATION_PUBLISHER = re.compile(
    r"(?:^|[.!?]\s+)[A-Z][^\n.:!?]{0,60}:\s*[A-Z][^\n.!?]{1,80}\.?\s*$"
)
# Recognise only nominal forms, not arbitrary short or title-cased strings. The
# closed suffix grammar deliberately leaves unknown headings alone: a layout
# label cannot establish that a sentence carries no claim.
_NOMINAL_HEADING_WORD = r"[^\W\d_]+(?:ation|ence|ity|ism)"
_NOMINAL_HEADING = re.compile(
    rf"(?:§\s*\d+[.)]?\s+|\d+[.)]?\s+)?"
    rf"{_NOMINAL_HEADING_WORD}"
    rf"(?:\s+(?:and|of|in|&|/)\s+(?:the\s+)?{_NOMINAL_HEADING_WORD})*",
    re.IGNORECASE | re.UNICODE,
)
_LOCATOR_NUMBER = re.compile(r"(?<![\w.])\d{1,4}(?:[–-]\d{1,4})?(?!\w)")

_FRONT_MATTER = re.compile(
    r"(?:ISBN|All rights reserved|Library of Congress|Cataloging|Cataloguing|"
    r"British Library|Printed (?:and bound )?in|First published|Typeset|"
    r"retrieval system|without (?:the )?prior (?:written )?permission|"
    r"Article reuse guidelines|Downloaded from|Terms and Conditions|"
    r"This content downloaded|All use subject to|Creative Commons|eBookstore)"
)
_FRONT_MATTER_MINIMUM_MARKERS = 2

# A sentence ends after a word of two or more letters, so initials hold together.
_ARGUMENT_SENTENCE_BREAK = re.compile(
    r"(?<=[^\W\d_]{2}[.!?])[\"'”’)]*\s+(?=[A-Z0-9“‘\"'(])"
)
_ARGUMENT_SENTENCE_MINIMUM_WORDS = 10
_ARGUMENT_GUARD_WORDS = 20
_ARGUMENT_GUARD_SHARE = 0.2

# A measurement is a thousands-grouped or decimal number, a signed value, or a
# percentage; a year, a page and a page range are not, and a link's digits are
# read away first.
_LINK = re.compile(r"(?:https?://|www\.|doi\.org/|\b10\.\d{4,9}/)\S+", re.IGNORECASE)
_MEASUREMENT = re.compile(
    r"(?<![\w.,])\d{1,3}(?:,\d{3})+(?![\d,])"
    r"|(?<![\w.])\d+\.\d+(?![\d.])"
    r"|(?<!\w)[−-]\d+(?:\.\d+)?"
    r"|(?<![\w.])\d+(?:\.\d+)?\s?%"
)
_TABLE_GUARD_MEASUREMENTS = 6


def _citation_density(value: str, words: int) -> float:
    markers = sum(
        len(pattern.findall(value))
        for pattern in (
            _CITATION_YEAR,
            _SURNAME_INITIALS,
            _INITIALS_SURNAME,
            _SURNAME_GIVEN,
            _CITATION_PAGES,
            _CITATION_LOCATOR,
            _CITATION_VENUE,
            _CITATION_NOTE,
        )
    )
    return markers / words


def _apparatus_shape(value: str, words: list[str], argument_share: float) -> str | None:
    count = max(len(words), 1)
    if argument_share <= _APPARATUS_MAXIMUM_ARGUMENT_SHARE:
        numbers = len(_LOCATOR_NUMBER.findall(value)) / count
        entries = len(_INDEX_ENTRY.findall(value))
        if (entries >= 3 and numbers >= 0.25) or (entries >= 1 and numbers >= 0.5):
            return APPARATUS_INDEX
        capitals = sum(word[:1].isupper() for word in words) / count
        if capitals >= _CITATION_MINIMUM_CAPITAL_SHARE and (
            _citation_density(value, count) >= _CITATION_MINIMUM_DENSITY
            or len(_CITATION_ENTRY.findall(value)) >= _CITATION_MINIMUM_ENTRIES
        ):
            return APPARATUS_BIBLIOGRAPHY
    if len(_FRONT_MATTER.findall(value)) >= _FRONT_MATTER_MINIMUM_MARKERS:
        return APPARATUS_FRONT_MATTER
    return None


def _is_argument_sentence(sentence: str, words: list[str]) -> bool:
    count = len(words)
    if count < _ARGUMENT_SENTENCE_MINIMUM_WORDS:
        return False
    capitals = sum(word[:1].isupper() for word in words)
    function_words = sum(word.casefold() in _FUNCTION_WORDS for word in words)
    numbers = len(_LOCATOR_NUMBER.findall(sentence))
    return capitals <= 0.3 * count and numbers < 0.2 * count and function_words >= 2


def _holds_evidence(value: str) -> bool:
    """Whether an apparatus-shaped passage still carries argument or data."""

    if len(_MEASUREMENT.findall(_LINK.sub(" ", value))) >= _TABLE_GUARD_MEASUREMENTS:
        return True
    total = argument = 0
    for sentence in _ARGUMENT_SENTENCE_BREAK.split(value.replace("\n", " ")):
        words = _WORD.findall(sentence)
        total += len(words)
        if _is_argument_sentence(sentence, words):
            argument += len(words)
    return argument >= _ARGUMENT_GUARD_WORDS or argument > _ARGUMENT_GUARD_SHARE * total


def _split_citation(value: str) -> bool:
    """An author-only line followed by a year-led bibliographic entry.

    Blank lines are layout, not a reason to lose the author signal. Do not change
    the source text or ingestion's stricter reference-removal rule.
    """

    lines = [line.strip() for line in value.splitlines() if line.strip()]
    return bool(
        len(lines) >= 2
        and _SPLIT_CITATION_AUTHOR.fullmatch(lines[0])
        and _SPLIT_CITATION_YEAR.match(lines[1])
        and (
            _CITATION_VENUE.search(value)
            or _CITATION_LOCATOR.search(value)
            or _CITATION_PUBLISHER.search(value)
        )
    )


def apparatus_kind(
    value: str, *, context: Mapping[str, object] | None = None
) -> str | None:
    """The kind of scholarly apparatus a passage is, or `None` for evidence.

    Returns `"bibliography"` for reference entries and citation-only notes,
    `"index"` for a back-of-book index, `"front_matter"` for a copyright page,
    or `"heading"` for a proven standalone nominal heading fragment. Short index
    entries need declared Index context. A heading label alone proves nothing;
    unknown forms and assertion-shaped headings remain eligible.
    """

    context = context or {}
    label = context.get("docling_label")
    if label is not None and label not in {
        "text",
        "reference",
        "list_item",
        "list",
        "section_header",
        "title",
        "footnote",
    }:
        return None
    if context.get("content_kind") in {"table", "formula", "caption", "code", "figure"}:
        return None
    if label in {"section_header", "title"}:
        return (
            APPARATUS_HEADING
            if context.get("whole_unit") is True
            and _NOMINAL_HEADING.fullmatch(normalize_inline_text(value))
            else None
        )
    words = _WORD.findall(value)
    argument_share = sum(word.casefold() in _ARGUMENT_WORDS for word in words) / max(
        len(words), 1
    )
    if argument_share > _FRONT_MATTER_MAXIMUM_ARGUMENT_SHARE:
        return None
    kind = _apparatus_shape(value, words, argument_share)
    if (
        kind is None
        and argument_share <= _APPARATUS_MAXIMUM_ARGUMENT_SHARE
        and context.get("section_kind") == APPARATUS_INDEX
        and _CONTEXTUAL_INDEX_ENTRY.fullmatch(value.strip())
    ):
        kind = APPARATUS_INDEX
    if kind is None and _split_citation(value):
        kind = APPARATUS_BIBLIOGRAPHY
    if kind is None or _holds_evidence(value):
        return None
    return kind


def chunk_health_flags(
    text: str,
    *,
    quality_flags: object = None,
    apparatus_context: Mapping[str, object] | None = None,
) -> int:
    """A precomputed verdict: an extraction artifact, no searchable alphanumeric
    content, or scholarly apparatus. Mirrors the query-time check, so counters
    cannot change.
    """
    flags = 0
    if text_corruption_reasons(text):
        flags |= CHUNK_FLAG_CORRUPT_TEXT
    stored = {str(item) for item in quality_flags or ()}
    if EXTRACTION_ARTIFACT_TOKEN in stored or not has_searchable_alphanumeric_content(
        text
    ):
        flags |= CHUNK_FLAG_EXTRACTION_ARTIFACT
    if apparatus_kind(text, context=apparatus_context) is not None:
        flags |= CHUNK_FLAG_APPARATUS
    return flags
