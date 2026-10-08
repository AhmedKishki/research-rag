"""Extracted text with its layout wrapping removed, one string at a time.

Every function here takes a string and returns a string. This module opens no
document, reads no file, and reaches no network, and it imports nothing from this
package.

Normalization never rewrites source prose. `clean_unclean_passage` is one
exception: it rewrites only a run of characters that were decoded with the wrong
single-byte codec, so its repair is exact and invents nothing. `recover_formatting_glyphs`
is the other: it decodes a glyph a recognised symbol face exposed as private use,
using Adobe's published encoding, and removes the non-prose glyphs of a recognised
icon or mathematics face. `text_quality` decides whether the result is usable, and
`extraction` assembles it into PDF and EPUB units.
"""

from __future__ import annotations

import re
import unicodedata

HORIZONTAL_SPACE = re.compile(r"[\t\f\v \u00a0]+")
LIST_ITEM_PATTERN = re.compile(r"^(?:[-*•]|\d+[.)]|[A-Za-z][.)])\s+")
_CONTROL_CHARACTERS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
# Compatibility folding is limited to the two blocks English scholarship
# produces: Mathematical Alphanumeric Symbols (letters from formula fonts,
# otherwise unmatchable by typed queries) and Alphabetic Presentation Forms
# (fi/fl/ff ligatures). Global NFKC would also fold superscripts, subscripts,
# and symbols that carry meaning in citations.
_FOLDABLE_CHARACTERS = re.compile(r"[\U0001d400-\U0001d7ff\ufb00-\ufb06\ufb13-\ufb17]")


def _starts_with_alpha(value: str) -> bool:
    for character in value:
        if character.isalpha():
            return True
        if character.isdigit():
            return False
    return False


def ends_sentence(value: str) -> bool:
    return value.rstrip("\"'”’)]}").endswith((".", "!", "?", "…", ":"))


def is_control_character(character: str) -> bool:
    """Whether a character is a C0 control or DEL this module strips."""

    return bool(_CONTROL_CHARACTERS.fullmatch(character))


# ---------------------------------------------------------------------------
# Recovering a formatting glyph a symbol font exposed as private use
# ---------------------------------------------------------------------------
#
# A PDF whose font carries no usable ToUnicode can expose one glyph per code as
# the private-use code point `0xF000 + code`. A Symbol face's bullet therefore
# arrives as U+F0B7 rather than U+2022, and a private-use code point is what the
# quality rules count as unreadable. The mapping below is Adobe's published
# Symbol encoding, so the recovery is the glyph the font actually drew and no
# guess. It is applied only to a span whose own font is a symbol or dingbat
# face: the same code point is a different glyph in an icon or mathematics font,
# and a private-use code point from any other face stays unreadable rather than
# being trusted as prose.
_ADOBE_SYMBOL_ENCODING: dict[int, str] = {
    0x20: " ",
    0x21: "!",
    0x22: "∀",
    0x23: "#",
    0x24: "∃",
    0x25: "%",
    0x26: "&",
    0x27: "∋",
    0x28: "(",
    0x29: ")",
    0x2A: "∗",
    0x2B: "+",
    0x2C: ",",
    0x2D: "−",
    0x2E: ".",
    0x2F: "/",
    0x30: "0",
    0x31: "1",
    0x32: "2",
    0x33: "3",
    0x34: "4",
    0x35: "5",
    0x36: "6",
    0x37: "7",
    0x38: "8",
    0x39: "9",
    0x3A: ":",
    0x3B: ";",
    0x3C: "<",
    0x3D: "=",
    0x3E: ">",
    0x3F: "?",
    0x40: "≅",
    0x41: "Α",
    0x42: "Β",
    0x43: "Χ",
    0x44: "Δ",
    0x45: "Ε",
    0x46: "Φ",
    0x47: "Γ",
    0x48: "Η",
    0x49: "Ι",
    0x4A: "ϑ",
    0x4B: "Κ",
    0x4C: "Λ",
    0x4D: "Μ",
    0x4E: "Ν",
    0x4F: "Ο",
    0x50: "Π",
    0x51: "Θ",
    0x52: "Ρ",
    0x53: "Σ",
    0x54: "Τ",
    0x55: "Υ",
    0x56: "ς",
    0x57: "Ω",
    0x58: "Ξ",
    0x59: "Ψ",
    0x5A: "Ζ",
    0x5B: "[",
    0x5C: "∴",
    0x5D: "]",
    0x5E: "⊥",
    0x5F: "_",
    0x61: "α",
    0x62: "β",
    0x63: "χ",
    0x64: "δ",
    0x65: "ε",
    0x66: "φ",
    0x67: "γ",
    0x68: "η",
    0x69: "ι",
    0x6A: "ϕ",
    0x6B: "κ",
    0x6C: "λ",
    0x6D: "µ",
    0x6E: "ν",
    0x6F: "ο",
    0x70: "π",
    0x71: "θ",
    0x72: "ρ",
    0x73: "σ",
    0x74: "τ",
    0x75: "υ",
    0x76: "ϖ",
    0x77: "ω",
    0x78: "ξ",
    0x79: "ψ",
    0x7A: "ζ",
    0x7B: "{",
    0x7C: "|",
    0x7D: "}",
    0x7E: "∼",
    0xA0: "€",
    0xA1: "ϒ",
    0xA2: "′",
    0xA3: "≤",
    0xA4: "⁄",
    0xA5: "∞",
    0xA6: "ƒ",
    0xA7: "♣",
    0xA8: "♦",
    0xA9: "♥",
    0xAA: "♠",
    0xAB: "↔",
    0xAC: "←",
    0xAD: "↑",
    0xAE: "→",
    0xAF: "↓",
    0xB0: "°",
    0xB1: "±",
    0xB2: "″",
    0xB3: "≥",
    0xB4: "×",
    0xB5: "∝",
    0xB6: "∂",
    0xB7: "•",
    0xB8: "÷",
    0xB9: "≠",
    0xBA: "≡",
    0xBB: "≈",
    0xBC: "…",
    0xBF: "↵",
    0xC0: "ℵ",
    0xC1: "ℑ",
    0xC2: "ℜ",
    0xC3: "℘",
    0xC4: "⊗",
    0xC5: "⊕",
    0xC6: "∅",
    0xC7: "∩",
    0xC8: "∪",
    0xC9: "⊃",
    0xCA: "⊇",
    0xCB: "⊄",
    0xCC: "⊂",
    0xCD: "⊆",
    0xCE: "∈",
    0xCF: "∉",
    0xD0: "∠",
    0xD1: "∇",
    0xD2: "®",
    0xD3: "©",
    0xD4: "™",
    0xD5: "∏",
    0xD6: "√",
    0xD7: "⋅",
    0xD8: "¬",
    0xD9: "∧",
    0xDA: "∨",
    0xDB: "⇔",
    0xDC: "⇐",
    0xDD: "⇑",
    0xDE: "⇒",
    0xDF: "⇓",
    0xE0: "◊",
    0xE1: "〈",
    0xE5: "∑",
    0xF1: "〉",
    0xF2: "∫",
    0xF3: "⌠",
    0xF5: "⌡",
}

# The codes Adobe's own table assigns to a Corporate Use Subarea code point
# because Unicode has no character for them. They are the pieces of a tall
# delimiter, so a PDF exposes them directly and they name no character to map to.
_SYMBOL_EXTENDERS = frozenset(range(0xF8E5, 0xF900))

# The Symbol codes that draw a formatting or mathematical symbol rather than a
# letter or a digit. A private-use code point in this set is a formatting glyph
# wherever it appears, so it never withholds a passage on its own. The letters
# and digits are left out on purpose: a private-use code point that stands for
# one of those is not evidence that a passage is readable.
_SYMBOL_FORMATTING_CODES = frozenset(
    code
    for code, character in _ADOBE_SYMBOL_ENCODING.items()
    if not character.isalnum()
)


def is_known_formatting_glyph(character: str) -> bool:
    """Whether a private-use code point is a documented formatting glyph.

    Covers the Symbol encoding's symbol codes and Adobe's Corporate Use Subarea
    extenders. It says nothing about a private-use code point from another font,
    which stays unreadable.
    """

    code = ord(character)
    if 0xF020 <= code <= 0xF0FF and (code - 0xF000) in _SYMBOL_FORMATTING_CODES:
        return True
    return code in _SYMBOL_EXTENDERS


def _formatting_font_kind(font: str) -> str:
    """The glyph-recovery family a font name names, or an empty string."""

    name = font.casefold()
    if "symbol" in name:
        return "symbol"
    if "dingbat" in name:
        return "dingbat"
    if any(
        marker in name
        for marker in ("fontawesome", "materialicons", "glyphicons", "icomoon")
    ):
        return "icon"
    if any(marker in name for marker in ("cmex", "cmsy", "msam", "msbm")):
        return "math"
    return ""


def recover_formatting_glyphs(value: str, font: str = "") -> str:
    """Recover the formatting glyphs a symbol face exposed as private use.

    A Symbol face is decoded with Adobe's published encoding, so U+F0B7 becomes
    the bullet it drew and U+F031 becomes the digit. A recognised dingbat, icon,
    or mathematics face draws no prose, so its private-use glyphs are removed
    rather than guessed at, which keeps the prose around them. Text from any
    other face, and every non-private character, is returned unchanged; nothing
    is invented and no missing glyph is supplied.
    """

    kind = _formatting_font_kind(font)
    if not kind:
        return value
    recovered: list[str] = []
    for character in value:
        if unicodedata.category(character) != "Co":
            recovered.append(character)
            continue
        code = ord(character)
        if kind == "symbol" and 0xF020 <= code <= 0xF0FF:
            mapped = _ADOBE_SYMBOL_ENCODING.get(code - 0xF000)
            if mapped is not None:
                recovered.append(mapped)
                continue
        # A recognised symbol, dingbat, icon, or mathematics face carries no
        # prose in this glyph. Drop it instead of guessing a character; a space
        # keeps the words on either side from joining.
        recovered.append(" ")
    return "".join(recovered)


def _fold_compatibility_characters(value: str) -> str:
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
        line = HORIZONTAL_SPACE.sub(" ", raw_line).strip()
        # PDF text spans sometimes leave a layout-only space after a hyphen
        # even though the printed form is a normal hyphenated word.
        line = re.sub(r"(?<=[^\W\d_])-\s+(?=[^\W\d_])", "-", line)
        if not line:
            separated = bool(current)
            continue
        if not current:
            current = line
        elif LIST_ITEM_PATTERN.match(line) or (separated and ends_sentence(current)):
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
    return HORIZONTAL_SPACE.sub(
        " ", normalize_reading_text(value).replace("\n", " ")
    ).strip()


# ---------------------------------------------------------------------------
# Repairing text a single-byte codec damaged on the way in
# ---------------------------------------------------------------------------
#
# A UTF-8 stream read as Latin-1 or Windows-1252 leaves one character per byte,
# so the damage is exactly reversible when the bytes are known. A character maps
# back to a byte only if it is a Latin-1 supplement character, whose code point
# is the byte, or one of the printable characters Windows-1252 puts where Latin-1
# has an unused control. Everything else — ASCII, non-Latin scripts, emoji —
# maps to nothing and ends the run, so clean and foreign text beside the damage
# is never part of what is rewritten.

# The bytes Windows-1252 assigns over Latin-1's C1 controls. Bytes 0x81, 0x8D,
# 0x8F, 0x90, and 0x9D have no character there: a stream holding one arrives as
# U+FFFD, which maps to nothing and is retained rather than guessed.
_CP1252_BYTE_BY_CHARACTER = {
    "\u20ac": 0x80,
    "\u201a": 0x82,
    "\u0192": 0x83,
    "\u201e": 0x84,
    "\u2026": 0x85,
    "\u2020": 0x86,
    "\u2021": 0x87,
    "\u02c6": 0x88,
    "\u2030": 0x89,
    "\u0160": 0x8A,
    "\u2039": 0x8B,
    "\u0152": 0x8C,
    "\u017d": 0x8E,
    "\u2018": 0x91,
    "\u2019": 0x92,
    "\u201c": 0x93,
    "\u201d": 0x94,
    "\u2022": 0x95,
    "\u2013": 0x96,
    "\u2014": 0x97,
    "\u02dc": 0x98,
    "\u2122": 0x99,
    "\u0161": 0x9A,
    "\u203a": 0x9B,
    "\u0153": 0x9C,
    "\u017e": 0x9E,
    "\u0178": 0x9F,
}

# One pass undoes a single mis-decode. A second undoes one that was encoded and
# mis-decoded twice, which is what a text that passed through two such steps
# carries; the bound stops a run from being rewritten without end.
_MOJIBAKE_REPAIR_PASSES = 2


def _byte_for_mojibake_character(character: str) -> int | None:
    """The octet a mis-decoded character stands for, or `None` if it is not one."""

    codepoint = ord(character)
    if 0x80 <= codepoint <= 0xFF:
        return codepoint
    return _CP1252_BYTE_BY_CHARACTER.get(character)


def _repair_byte_run(run: str) -> str:
    """The longest UTF-8 decodings of one contiguous run of mis-decoded bytes.

    A run may hold a valid sequence followed by bytes that are not part of one.
    The longest valid prefix is decoded and the process repeats on what follows,
    so a sequence that decodes is repaired and an incomplete tail is left as it
    arrived. A byte that decodes alone is never rewritten, because a repair
    replaces a run of bytes with the characters they encoded, not a lone byte
    with itself.
    """

    raw = bytes(_byte_for_mojibake_character(character) for character in run)
    repaired: list[str] = []
    index = 0
    end = len(raw)
    while index < end:
        decoded: str | None = None
        stop = index
        for candidate_end in range(min(index + 4, end), index, -1):
            try:
                decoded = raw[index:candidate_end].decode("utf-8")
            except UnicodeDecodeError:
                continue
            stop = candidate_end
            break
        if decoded is None or stop - index < 2:
            repaired.append(run[index])
            index += 1
            continue
        repaired.append(decoded)
        index = stop
    return "".join(repaired)


def _repair_mojibake(value: str) -> str:
    """Rewrite every run of mis-decoded bytes once, leaving the rest in place."""

    repaired: list[str] = []
    index = 0
    end = len(value)
    while index < end:
        if _byte_for_mojibake_character(value[index]) is None:
            repaired.append(value[index])
            index += 1
            continue
        start = index
        while index < end and _byte_for_mojibake_character(value[index]) is not None:
            index += 1
        repaired.append(_repair_byte_run(value[start:index]))
    return "".join(repaired)


def clean_unclean_passage(value: str) -> str:
    """Repair a passage a wrong single-byte codec damaged, or return it unchanged.

    Call this only for a passage `text_quality` has already judged unhealthy. It
    rewrites a run of characters back to the bytes they stand for and decodes
    those bytes as UTF-8, at most twice, and returns the existing normalization
    of a repaired string. A run that does not decode as UTF-8, and every part of
    the passage that is not such a run, is returned unchanged. A missing glyph
    is never guessed, a replacement character inside a word is never deleted, a
    foreign script is never transliterated, and no model, OCR, network, or
    subprocess is reached. Clean text is returned exactly as it arrived.
    """

    repaired = value
    for _ in range(_MOJIBAKE_REPAIR_PASSES):
        candidate = _repair_mojibake(repaired)
        if candidate == repaired:
            break
        repaired = candidate
    if repaired == value:
        return value
    return normalize_inline_text(repaired)
