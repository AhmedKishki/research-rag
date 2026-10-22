"""Load sources, COPY exact spans, and hold the deviation a sentence states.

Copied from github.com/AhmedKishki/agent-skills @ ea68182,
`skills/white-box-synthesis/scripts/ops.py`. The copy is deliberate; carry a fix
to both copies.
"""

import re

MARKER = re.compile(r"⟦[^⟧]*⟧|\[(?:GAP|AI[ -]?authored)\b[^\]]*\]", re.IGNORECASE)
QUOTES = str.maketrans("‘’“”", "''\"\"")
# The deviation the sentence being built states. A form rule it breaks is recorded
# here instead of refused; the rules on copying and invention never read it.
WAIVER = {"reason": None, "waived": []}


class Rejected(ValueError):
    """A derivation step that the rules do not license."""


def guideline(message):
    """Refuse a form-rule break unless the sentence being built states why."""
    if not WAIVER["reason"]:
        raise Rejected(
            f"{message} Fix it, or give the sentence a deviation saying why."
        )
    WAIVER["waived"].append(message)


def is_word(ch):
    return ch.isalnum() or ch == "_"


def find(text, needle, occurrence=1, what="span"):
    """Return (start, end) of a whole-word occurrence of needle in text."""
    start = -1
    for _ in range(occurrence):
        start = text.find(needle, start + 1) if needle else -1
        if start < 0:
            near = needle and needle.translate(QUOTES) in text.translate(QUOTES)
            hint = " A match ignoring quote glyphs exists." if near else ""
            raise Rejected(f"{what} {needle!r} not found.{hint}")
    end = start + len(needle)
    for i in (start, end):
        if 0 < i < len(text) and is_word(text[i - 1]) and is_word(text[i]):
            guideline(f"{what} {needle!r} splits a word.")
            break
    return start, end


def load_sources(raw):
    if not isinstance(raw, dict) or not raw:
        raise Rejected("sources must be a non-empty object keyed by label.")
    out = {}
    for label, meta in raw.items():
        text = re.sub(r"\s+", " ", str(meta.get("text", ""))).strip()
        if not text:
            raise Rejected(f"{label}: source text is empty.")
        clean = MARKER.sub(lambda m: "\0" * len(m.group()), text)
        out[label] = dict(meta, text=text, clean=clean)
    return out


def copy_span(sources, label, text, occurrence=1, within=None):
    """Locate an exact span. Spans inside AI-authored or gap markers are refused."""
    if label not in sources:
        raise Rejected(f"Unknown source {label!r}.")
    src = sources[label]
    lo, hi = within or (0, len(src["text"]))
    start, end = find(src["text"][lo:hi], text, occurrence, f"{label} COPY")
    start, end = start + lo, end + lo
    if "\0" in src["clean"][start:end]:
        raise Rejected(f"{label}: {text!r} overlaps an AI-authored or gap marker.")
    return start, end
