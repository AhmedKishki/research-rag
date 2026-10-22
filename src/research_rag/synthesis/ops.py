"""Replay DELETE, INFLECT and NORMALISE on exact source spans.

Copied from github.com/AhmedKishki/agent-skills @ a08197f,
`skills/white-box-synthesis/scripts/ops.py`. The copy is deliberate; carry a fix
to both copies.
"""

import re

MARKER = re.compile(r"⟦[^⟧]*⟧|\[(?:GAP|AI[ -]?authored)\b[^\]]*\]", re.IGNORECASE)
QUOTES = str.maketrans("‘’“”", "''\"\"")
PAIRS = {
    ("is", "are"),
    ("was", "were"),
    ("has", "have"),
    ("does", "do"),
    ("this", "these"),
    ("that", "those"),
    ("a", "an"),
}
PRONOUNS = {
    "he",
    "she",
    "it",
    "they",
    "him",
    "her",
    "them",
    "this",
    "these",
    "that",
    "those",
    "its",
    "their",
    "his",
    "hers",
    "theirs",
}
POSSESSIVE = {"its", "their", "his", "her", "hers", "theirs"}


class Rejected(ValueError):
    """A derivation step that the closed rules do not license."""


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
            raise Rejected(f"{what} {needle!r} splits a word.")
    return start, end


def agrees(old, new):
    """True when old -> new is an agreement change on the closed list."""
    a, b = old.lower(), new.lower()
    if old[:1].isupper() != new[:1].isupper():
        return False
    for x, y in ((a, b), (b, a)):
        if (x, y) in PAIRS or y in (x + "s", x + "es"):
            return True
        if x.endswith("y") and y == x[:-1] + "ies":
            return True
    return False


def inflect(text, op, ctx):
    kind, old, new = op.get("kind"), op.get("from", ""), op.get("to", "")
    start, end = find(text, old, op.get("occurrence", 1), "INFLECT from")
    if kind == "agreement":
        if not agrees(old, new):
            raise Rejected(f"INFLECT {old!r} -> {new!r} is not on the agreement list.")
    elif kind == "antecedent":
        if old.lower() not in PRONOUNS:
            raise Rejected(f"INFLECT antecedent: {old!r} is not a listed pronoun.")
        ends = ("'s", "’s", "'", "’", "") if old.lower() in POSSESSIVE else ("",)
        bases = [new[: len(new) - len(s)] for s in ends if s and new.endswith(s)]
        if not any(b and b in ctx["source"] for b in bases + [new]):
            raise Rejected(
                f"INFLECT antecedent {new!r} is not copied from the same source."
            )
    else:
        raise Rejected(
            f"INFLECT kind {kind!r} is not allowed (agreement or antecedent)."
        )
    return text[:start] + new + text[end:], f'INFLECT {kind} "{old}" → "{new}"'


def normalise(text, op, ctx):
    rule = op.get("rule")
    if rule == "case":
        i = next((k for k, ch in enumerate(text) if ch.isalpha()), None)
        if i is None or op.get("to") not in ("upper", "lower"):
            raise Rejected("NORMALISE case needs a letter and to: upper or lower.")
        ch = text[i].upper() if op["to"] == "upper" else text[i].lower()
        return text[:i] + ch + text[i + 1 :], f'NORMALISE case "{text[i]}" → "{ch}"'
    if rule == "quotes":
        result = op.get("result", "")
        if result.translate(QUOTES) != text.translate(QUOTES):
            raise Rejected("NORMALISE quotes may change quote glyphs only.")
        return result, "NORMALISE quote glyphs"
    if rule == "end-stop":
        if not ctx["declarative"]:
            raise Rejected(
                "NORMALISE end-stop: the source sentence does not end with a full stop."
            )
        return text.rstrip(",;: ") + ".", "NORMALISE end-stop"
    if rule == "semicolon":
        start, end = find(text, ";", op.get("occurrence", 1), "NORMALISE semicolon")
        rest = text[end:].lstrip()
        if not rest[:1].isalpha():
            raise Rejected("NORMALISE semicolon must be followed by a clause.")
        return text[:start] + ". " + rest[0].upper() + rest[
            1:
        ], "NORMALISE semicolon → full stop"
    if rule == "pair":
        old, new = op.get("from", ""), op.get("to", "")
        if {"from": old, "to": new} not in ctx["pairs"]:
            raise Rejected(f"NORMALISE pair {old!r} -> {new!r} is not declared.")
        start, end = find(text, old, op.get("occurrence", 1), "NORMALISE pair")
        return text[:start] + new + text[end:], f'NORMALISE "{old}" → "{new}"'
    raise Rejected(f"NORMALISE rule {rule!r} is not on the closed list.")
