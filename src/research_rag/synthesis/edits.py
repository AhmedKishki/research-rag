"""INFLECT and NORMALISE on their closed lists. Neither adds a word no source wrote.

Copied from github.com/AhmedKishki/agent-skills @ 7dfca79,
`skills/white-box-synthesis/scripts/edits.py`. The copy is deliberate; carry a fix
to both copies.
"""

import re

from .ops import QUOTES, Rejected, find, guideline

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
        if re.sub(r"[\W_]", "", result) != re.sub(r"[\W_]", "", text):
            raise Rejected(
                "NORMALISE quotes may change punctuation only, never a letter."
            )
        if result.translate(QUOTES) != text.translate(QUOTES):
            guideline("NORMALISE quotes changes more than quote glyphs.")
        return result, "NORMALISE quote glyphs"
    if rule == "end-stop":
        if not ctx["declarative"]:
            guideline(
                "NORMALISE end-stop: the source sentence does not end with a full stop."
            )
        return text.rstrip(",;: ") + ".", "NORMALISE end-stop"
    if rule == "semicolon":
        start, end = find(text, ";", op.get("occurrence", 1), "NORMALISE semicolon")
        rest = text[end:].lstrip()
        if not rest[:1].isalpha():
            guideline("NORMALISE semicolon must be followed by a clause.")
        return text[:start] + ". " + rest[:1].upper() + rest[
            1:
        ], "NORMALISE semicolon → full stop"
    if rule in ("pair", "space"):
        old, new = op.get("from", ""), op.get("to", "")
        if rule == "pair" and {"from": old, "to": new} not in ctx["pairs"]:
            raise Rejected(f"NORMALISE pair {old!r} -> {new!r} is not declared.")
        if rule == "space" and not " ".join(new.split()) == new != old == new.replace(
            " ", ""
        ):
            raise Rejected(
                f"NORMALISE space {old!r} -> {new!r} may only insert spaces."
            )
        start, end = find(text, old, op.get("occurrence", 1), f"NORMALISE {rule}")
        return text[:start] + new + text[end:], f'NORMALISE {rule} "{old}" → "{new}"'
    raise Rejected(f"NORMALISE rule {rule!r} is not on the closed list.")
