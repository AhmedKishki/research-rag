"""Replay each piece's operations and join pieces into sentences.

Copied from github.com/AhmedKishki/agent-skills @ ea68182,
`skills/white-box-synthesis/scripts/pieces.py`. The copy is deliberate; carry a fix
to both copies.
"""

import re

from .edits import inflect, normalise
from .ops import WAIVER, Rejected, copy_span, find, guideline, is_word


def replay(piece, sources, pairs, within=None):
    """COPY one piece and apply its operations in order."""
    label, text = piece.get("source"), piece.get("text", "")
    start, end = copy_span(sources, label, text, piece.get("occurrence", 1), within)
    src = sources[label]
    stop = re.search(r"[.?!]", src["text"][end - 1 :])
    ctx = {
        "source": src["clean"],
        "pairs": pairs,
        "declarative": bool(stop) and stop.group() == ".",
    }
    notes = []
    for op in piece.get("ops", []):
        kind = op.get("op")
        if kind == "DELETE":
            s, e = find(text, op.get("text", ""), op.get("occurrence", 1), "DELETE")
            text = text[:s] + text[e:]
            if 0 < s < len(text) and is_word(text[s - 1]) and is_word(text[s]):
                guideline(f"DELETE {op['text']!r} would merge two words.")
            notes.append(f'DELETE "{op["text"]}"')
        elif kind == "INFLECT":
            text, note = inflect(text, op, ctx)
            notes.append(note)
        elif kind == "NORMALISE":
            text, note = normalise(text, op, ctx)
            notes.append(note)
        else:
            raise Rejected(
                f"Unknown operation {kind!r}. Use DELETE, INFLECT or NORMALISE."
            )
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        raise Rejected(f"{label}: operations leave no wording.")
    return text, label, (start, end), notes


def build_sentence(sentence, steps, sources, pairs):
    """Return (text, labels, provenance, spans, waived): the rules its deviation waived."""
    WAIVER.update(reason=sentence.get("deviation"), waived=[])
    try:
        return join_pieces(sentence, steps, sources, pairs) + (WAIVER["waived"],)
    finally:
        WAIVER.update(reason=None, waived=[])


def join_pieces(sentence, steps, sources, pairs):
    """Join the pieces of one sentence. Pieces join through a licensed connective."""
    pieces = sentence.get("pieces") or []
    texts, labels, prov, spans, last_content = [], [], [], [], None
    for i, piece in enumerate(pieces):
        link, within, tag = piece.get("link"), None, ""
        if link is not None:
            step = steps.get(link) or {}
            lic = step.get("licence") or {}
            if step.get("status") != "stated":
                guideline(
                    f"Connective {piece.get('text')!r} needs a stated link on {link!r}."
                )
            elif piece.get("source") != lic.get("source"):
                guideline("A connective must be copied from its licence's source.")
            else:
                within = copy_span(
                    sources,
                    lic["source"],
                    lic.get("text", ""),
                    lic.get("occurrence", 1),
                )
            tag = f" connective for {link} → {step.get('supports')}"
        else:
            if last_content == i - 1:
                guideline(
                    "Two content pieces are joined without a licensed connective."
                )
            last_content = i
        text, label, span, notes = replay(piece, sources, pairs, within)
        texts.append(text)
        spans.append((label, span))
        labels += [label] if label not in labels else []
        prov.append(f"{label}{tag}: " + (" · ".join(notes) or "verbatim"))
    if last_content is None:
        raise Rejected("A sentence needs at least one content piece.")
    out = ""
    for text in texts:
        out += ("" if not out or re.match(r"[,.;:!?)\]”’]", text) else " ") + text
    if (lead := re.search(r"[^\W\d_]", out)) and lead.group().islower():
        guideline(f"{out[:40]!r} must start with a capital: NORMALISE case to upper.")
    mark = "[" + "; ".join(labels) + "]"
    end = re.search(r"[.?!][”’\"')\]]*$", out)
    if not end:
        guideline(f"{out[-40:]!r} must end with a full stop: NORMALISE end-stop.")
    out = f"{out[: end.start()]} {mark}{end.group()}" if end else f"{out} {mark}"
    return out, labels, " + ".join(prov), spans
