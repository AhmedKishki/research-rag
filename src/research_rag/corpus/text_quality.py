"""Forensics on one string of extracted text, and the verdicts built from them.

A reason is evidence about the text itself, never about the source it came from.
This module reads no file, opens no document, reaches no network, and imports
nothing from this package.

`text_normalization` owns the text every verdict is computed over. `extraction`
withholds a unit whose text carries a reason.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter

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
