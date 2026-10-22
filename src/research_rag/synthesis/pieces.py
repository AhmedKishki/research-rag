"""COPY source spans, replay their operations and join pieces into sentences.

Copied from github.com/AhmedKishki/agent-skills @ d7d06da,
`skills/white-box-synthesis/scripts/pieces.py`. The copy is deliberate; carry a fix
to both copies.
"""

import re

from .ops import MARKER, Rejected, find, inflect, is_word, normalise


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
                raise Rejected(f"DELETE {op['text']!r} would merge two words.")
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
    """Return (text, labels, provenance, spans). Joins need a licensed connective."""
    pieces = sentence.get("pieces") or []
    texts, labels, prov, spans, last_content = [], [], [], [], None
    for i, piece in enumerate(pieces):
        link, within, tag = piece.get("link"), None, ""
        if link is not None:
            step = steps.get(link) or {}
            lic = step.get("licence") or {}
            if step.get("status") != "stated":
                raise Rejected(
                    f"Connective {piece.get('text')!r} needs a stated link on {link!r}."
                )
            if piece.get("source") != lic.get("source"):
                raise Rejected("A connective must be copied from its licence's source.")
            within = copy_span(
                sources, lic["source"], lic.get("text", ""), lic.get("occurrence", 1)
            )
            tag = f" connective for {link} → {step['supports']}"
        else:
            if last_content == i - 1:
                raise Rejected(
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
        raise Rejected(
            f"{out[:40]!r} must start with a capital: NORMALISE case to upper."
        )
    mark = "[" + "; ".join(labels) + "]"
    end = re.search(r"[.?!][”’\"')\]]*$", out)
    out = f"{out[: end.start()]} {mark}{end.group()}" if end else f"{out} {mark}"
    return out, labels, " + ".join(prov), spans
