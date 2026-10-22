"""Render a white-box derivation: Synthesis, References, Provenance, Reasoning.

Copied from github.com/AhmedKishki/agent-skills @ ea68182,
`skills/white-box-synthesis/scripts/render.py`. The copy is deliberate: this app
depends on no other project by its author. Carry a fix to both copies.

Here the script's command-line entry is `render`, which refuses a malformed
derivation as the script's exit 1 does, and `render_parts`, which keeps the four
parts apart for a caller that lays them out itself. `render_parts` also names
each sentence and each deviation, so the app can hold a sentence to its own
rules and report what a deviation waived.
"""

import re

from .checks import check_argument, check_sources
from .ops import Rejected, load_sources
from .pieces import build_sentence
from .reasoning import reasoning, reference

PARTS = ("synthesis", "references", "provenance", "reasoning")

# What a derivation written by hand or by a model can raise before a rule
# names its fault. The script's entry reads the same set as a rejection.
_MALFORMED = (KeyError, TypeError, IndexError, AttributeError)


def link_text(step):
    kind = {"gap": ("GAP", "missing"), "ai": ("AI-AUTHORED", "text")}.get(
        step.get("status")
    )
    return f"⟦{kind[0]}: {step[kind[1]]}⟧" if kind else None


def render(d):
    """Return the four parts as one Markdown reply, or raise `Rejected`."""

    parts = render_parts(d)
    titles = dict(zip(PARTS, ("Synthesis", "References", "Provenance", "Reasoning")))
    return "\n\n".join(f"**{titles[p]}**\n\n{parts[p]}" for p in PARTS) + "\n"


def render_parts(d):
    """Return the four parts, `deviations` and `sentences`, or raise `Rejected`."""

    try:
        return _render(d)
    except _MALFORMED as error:
        raise Rejected(f"The derivation is malformed: {error!r}.") from error


def _render(d):
    sources = load_sources(d.get("sources"))
    steps = check_argument(d, sources)
    results = check_sources(d, sources)
    order, built, last, cursor = list(steps), [], {}, -1
    pairs = d.get("normalisations", [])
    for n, sentence in enumerate(d.get("sentences") or [], 1):
        ids = sentence.get("step")
        ids = ids if isinstance(ids, list) else [ids]
        if any(i not in steps for i in ids):
            raise Rejected(f"Sentence {n}: unknown step {ids}.")
        late = (k := min(order.index(i) for i in ids)) < cursor
        if late and not sentence.get("deviation"):
            raise Rejected(
                f"Sentence {n}: sentences must follow the chain order of steps. "
                "Fix it, or give the sentence a deviation saying why."
            )
        cursor = max(cursor, k)
        text, labels, line, used, waived = build_sentence(
            sentence, steps, sources, pairs
        )
        waived = ["Sentences must follow the chain order of steps."] * late + waived
        built.append((n, sentence, ids, text, labels, line, used, waived))
        last.update({i: n for i in ids})
    if missing := [i for i in steps if i not in last]:
        raise Rejected(
            f"No source sentence for {', '.join(missing)}: drop it and mark a gap."
        )
    paras, prov, refs, spans = [[]], [], [], {}
    for n, sentence, ids, text, labels, line, used, waived in built:
        if sentence.get("paragraph") and paras[-1]:
            paras.append([])
        paras[-1].append(text)
        why = sentence.get("deviation")
        prov.append(f"{n}. {line}" + (f" · DEVIATION: {why}" if why else ""))
        refs += labels
        for label, span in used:
            spans.setdefault(label, []).append(span)
        for sid in (i for i in steps if last[i] == n and link_text(steps[i])):
            paras[-1].append(mark := link_text(steps[sid]))
            refs += [r for r in sources if re.search(rf"\b{re.escape(r)}\b", mark)]
            kind = steps[sid]["status"].upper()
            prov.append(f"- After {n}: {kind} for {sid} → {steps[sid]['supports']}")
    return {
        "synthesis": "\n\n".join(" ".join(p) for p in paras),
        "references": "\n".join(reference(sources[r], r) for r in dict.fromkeys(refs)),
        "provenance": "\n".join(prov),
        "reasoning": "\n".join(reasoning(d, steps, sources, results, spans, built)),
        "deviations": [
            {"sentence": n, "reason": s["deviation"], "waived": w}
            for n, s, *_, w in built
            if s.get("deviation")
        ],
        "sentences": [
            {"sentence": n, "text": text, "deviation": s.get("deviation") or None}
            for n, s, _, text, *_ in built
        ],
    }
