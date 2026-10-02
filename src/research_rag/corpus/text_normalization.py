"""Extracted text with its layout wrapping removed, one string at a time.

Every function here takes a string and returns a string. This module opens no
document, reads no file, and reaches no network, and it imports nothing from this
package.

Normalization never rewrites source prose. `text_quality` decides whether the
result is usable, and `extraction` assembles it into PDF and EPUB units.
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
